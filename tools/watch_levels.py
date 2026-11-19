"""Read-only price watcher (public MEXC data, no keys, no orders). Publishes events to branch `watch-results`.
Usage: python tools/watch_levels.py SYMBOL LEVELS_JSON   e.g. PIEVERSE_USDT '{"tp":[3.37,3.53,3.63],"stop":3.0,"long":true}'"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

POLL, HEARTBEAT, MAX_SEC, REMIND_SEC = 20, 600, 350 * 60, 7200


def check(levels, hit, hi, lo):
    """Return new event strings for candle extremes (hi, lo). `hit` is a set that is updated."""
    ev, long_ = [], levels.get("long", True)
    up, dn = (hi, lo) if long_ else (lo, hi)
    for i, tp in enumerate(levels["tp"], 1):
        if f"TP{i}" not in hit and ((up >= tp) if long_ else (up <= tp)):
            hit.add(f"TP{i}"); ev.append(f"TP{i} بلغ {tp}")
    st = levels["stop"]
    if "STOP" not in hit and ((dn <= st) if long_ else (dn >= st)):
        hit.add("STOP"); ev.append(f"الوقف لُمس عند {st}")
    return ev


def trail_step(hh, stop, hi, lo, risk, stop0):
    """One poll of a long trailing stop. -> (hit, new_stop, new_highest). The stop only rises; it is checked BEFORE this window's high is applied."""
    if lo <= stop:
        return True, stop, hh
    hh = max(hh, hi)
    return False, max(stop, stop0, hh - risk), hh


def publish(events, status, repo_dir="/tmp/watch_out"):
    d = Path(repo_dir); d.mkdir(exist_ok=True)
    (d / "events.txt").write_text("\n".join(events) + "\n", encoding="utf-8")
    n0 = getattr(publish, "n", 0)
    if len(events) > n0:                                               # keep every event across restarts
        with (d / "history.txt").open("a", encoding="utf-8") as f:
            f.write("\n".join(events[n0:]) + "\n")
        publish.n = len(events)
    (d / "status.txt").write_text(status + "\n", encoding="utf-8")
    run = lambda *a: subprocess.run(a, cwd=d, check=False, capture_output=True, text=True)
    if not (d / ".git").exists():
        run("git", "init", "-q"); run("git", "checkout", "-q", "-b", "watch-results")
        run("git", "config", "user.name", "watch-bot"); run("git", "config", "user.email", "bot@example.com")
        run("git", "remote", "add", "origin", f"https://x-access-token:{os.environ['GH_TOKEN']}@github.com/{os.environ['GITHUB_REPOSITORY']}.git")
    run("git", "add", "-A"); run("git", "commit", "-q", "-m", "watch")
    for _ in range(3):
        if run("git", "push", "-q", "-f", "origin", "watch-results").returncode == 0:
            break
        time.sleep(3)


def main():
    from tradebot.config import load_config
    from tradebot.mexc.client import MexcClient

    arg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
    specs = json.loads(sys.argv[1]) if sys.argv[1].lstrip().startswith("[") else [dict(arg, sym=sys.argv[1])]
    cl = MexcClient(load_config("config.toml").mexc)
    t0, events, last_pub = time.time(), [], 0.0
    stamp = lambda: time.strftime("%H:%M:%S", time.gmtime()) + " UTC"
    st = {}
    for sp in specs:
        px = cl.last_price(sp["sym"])
        st[sp["sym"]] = dict(sp=sp, hit=set(), px=px, done=False, reminded=False, hh=max(px, sp.get("hh", 0)), stop=sp.get("stop0", 0), rep=sp.get("stop0", 0))
        if "trail" in sp:
            events.append(f"{stamp()} · بدأت مراقبة {sp['sym']} بوقف متحرك · السعر {px} · وقف أول {sp['stop0']} · مسافة الوقف المتحرك {sp['trail']} · الهدف مفتوح")
        else:
            events.append(f"{stamp()} · بدأت مراقبة {sp['sym']} · السعر {px} · أهداف {sp['tp']} · وقف {sp['stop']}")
    publish(events, f"{stamp()} " + " · ".join(f"{k} {v['px']}" for k, v in st.items()))
    while time.time() - t0 < MAX_SEC and not all(v["done"] for v in st.values()):
        pushed = False
        for sym, v in st.items():
            if v["done"]:
                continue
            try:
                now = int(time.time())
                k = cl.klines(sym, "Min1", now - 240, now)
                px = cl.last_price(sym)
                hi = max(list(k.get("high", [])) + [px]); lo = min(list(k.get("low", [])) + [px])
                v["px"] = px
                if "trail" in v["sp"]:                                   # open target + trailing stop
                    hit, v["stop"], v["hh"] = trail_step(v["hh"], v["stop"], hi, lo, v["sp"]["trail"], v["sp"]["stop0"])
                    if hit:
                        events.append(f"{stamp()} · {sym} · الوقف المتحرك لُمس عند {v['stop']:.6g} (أعلى سعر {v['hh']:.6g}) · السعر الآن {px}")
                        v["done"] = True; pushed = True
                    elif v["stop"] >= v["rep"] + 0.25 * v["sp"]["trail"]:
                        v["rep"] = v["stop"]; pushed = True
                        events.append(f"{stamp()} · {sym} · الوقف المتحرك ارتفع إلى {v['stop']:.6g} (أعلى سعر {v['hh']:.6g}) · السعر الآن {px}")
                    if not v["done"] and time.time() > v["sp"].get("deadline", 1e18):
                        events.append(f"{stamp()} · {sym} · انتهت مهلة الصفقة · السعر {px} · الوقف {v['stop']:.6g}"); v["done"] = True; pushed = True
                    continue
                new = check(v["sp"], v["hit"], hi, lo)
                for e in new:
                    events.append(f"{stamp()} · {sym} · {e} · السعر الآن {px}")
                if not v["reminded"] and time.time() - t0 >= REMIND_SEC and "TP1" not in v["hit"]:
                    v["reminded"] = True; new.append("reminder")
                    events.append(f"{stamp()} · {sym} · مرّت ساعتان دون بلوغ الهدف الأول · السعر {px}")
                pushed = pushed or bool(new)
                if "STOP" in v["hit"] or f"TP{len(v['sp']['tp'])}" in v["hit"]:
                    v["done"] = True; events.append(f"{stamp()} · {sym} · انتهت مراقبة هذه العملة")
                    pushed = True
            except Exception as e:                                    # transient API errors must not kill the watch
                print("err", sym, e, file=sys.stderr)
        if pushed or time.time() - last_pub >= HEARTBEAT:
            publish(events, f"{stamp()} " + " · ".join(f"{k} {v['px']} {sorted(v['hit']) or '-'} stop={v['stop']:.6g} hh={v['hh']:.6g}" for k, v in st.items())); last_pub = time.time()
        time.sleep(POLL)
    events.append(f"{stamp()} · انتهت المراقبة"); publish(events, f"{stamp()} انتهت")


if __name__ == "__main__":
    main()
