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


def publish(events, status, repo_dir="/tmp/watch_out"):
    d = Path(repo_dir); d.mkdir(exist_ok=True)
    (d / "events.txt").write_text("\n".join(events) + "\n", encoding="utf-8")
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

    sym, levels = sys.argv[1], json.loads(sys.argv[2])
    cl = MexcClient(load_config("config.toml").mexc)
    t0, hit, events, last_pub = time.time(), set(), [], 0.0
    stamp = lambda: time.strftime("%H:%M:%S", time.gmtime()) + " UTC"
    px = cl.last_price(sym)
    events.append(f"{stamp()} · بدأت مراقبة {sym} · السعر {px} · أهداف {levels['tp']} · وقف {levels['stop']}")
    publish(events, f"{stamp()} السعر {px}")
    reminded = False
    while time.time() - t0 < MAX_SEC:
        try:
            now = int(time.time())
            k = cl.klines(sym, "Min1", now - 240, now)
            hi = max(list(k.get("high", [])) or [px]); lo = min(list(k.get("low", [])) or [px])
            px = cl.last_price(sym)
            hi, lo = max(hi, px), min(lo, px)
            new = check(levels, hit, hi, lo)
            for e in new:
                events.append(f"{stamp()} · {sym} · {e} · السعر الآن {px}")
            if not reminded and time.time() - t0 >= REMIND_SEC and "TP1" not in hit:
                reminded = True; events.append(f"{stamp()} · {sym} · مرّت ساعتان دون بلوغ الهدف الأول · السعر {px}")
                new.append("reminder")
            if new or time.time() - last_pub >= HEARTBEAT:
                publish(events, f"{stamp()} السعر {px} · محقق: {sorted(hit) or 'لا شيء'}"); last_pub = time.time()
            if "STOP" in hit or f"TP{len(levels['tp'])}" in hit:
                events.append(f"{stamp()} · انتهت المراقبة"); publish(events, f"{stamp()} انتهت"); return
        except Exception as e:                                        # transient API errors must not kill the watch
            print("err", e, file=sys.stderr)
        time.sleep(POLL)
    events.append(f"{stamp()} · انتهت مهلة المراقبة (6 ساعات) · محقق: {sorted(hit) or 'لا شيء'}"); publish(events, f"{stamp()} انتهت المهلة")


if __name__ == "__main__":
    main()
