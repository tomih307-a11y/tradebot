"""MEXC Futures REST client (contract.mexc.com).

Public: klines, tickers, contract detail, funding-rate history.
Private (HMAC-SHA256): positions, orders, stop orders.

NOTE: the private endpoints below follow MEXC's published futures API but were
written without network access to MEXC, so they are UNVERIFIED against the live
exchange. Run with trading.dry_run = true first and check each call with a
tiny size before trusting them. All exchange-specific payloads live in this file.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request

log = logging.getLogger("mexc")

TF_SECONDS = {"Min1": 60, "Min5": 300, "Min15": 900, "Min30": 1800, "Min60": 3600,
              "Hour4": 14400, "Hour8": 28800, "Day1": 86400}

# order side codes
OPEN_LONG, CLOSE_SHORT, OPEN_SHORT, CLOSE_LONG = 1, 2, 3, 4


class MexcError(RuntimeError):
    pass


class MexcClient:
    def __init__(self, cfg_mexc):
        self.base = cfg_mexc.base_url.rstrip("/")
        self.timeout = cfg_mexc.request_timeout_sec
        self.retries = cfg_mexc.max_retries
        self.page_limit = cfg_mexc.kline_page_limit
        self.api_key = os.environ.get(cfg_mexc.api_key_env, "")
        self.api_secret = os.environ.get(cfg_mexc.api_secret_env, "")

    # ───────────── transport ─────────────
    def _request(self, method: str, path: str, params: dict | None = None,
                 body: dict | list | None = None, private: bool = False):
        params = {k: v for k, v in (params or {}).items() if v is not None}
        query = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in sorted(params.items()))
        url = self.base + path + (("?" + query) if query else "")
        data = None
        headers = {"Content-Type": "application/json"}
        if body is not None:
            data = json.dumps(body, separators=(",", ":")).encode()
        if private:
            if not (self.api_key and self.api_secret):
                raise MexcError("MEXC API keys missing (env vars) for private call")
            ts = str(int(time.time() * 1000))
            payload = data.decode() if data is not None else query
            sig = hmac.new(self.api_secret.encode(), (self.api_key + ts + payload).encode(),
                           hashlib.sha256).hexdigest()
            headers.update({"ApiKey": self.api_key, "Request-Time": ts, "Signature": sig})
        last = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method=method)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read().decode())
                if isinstance(out, dict) and out.get("success") is False:
                    raise MexcError(f"{path}: code={out.get('code')} msg={out.get('message')}")
                return out.get("data") if isinstance(out, dict) and "data" in out else out
            except MexcError:
                raise
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last = e
                time.sleep(min(2 ** attempt, 8))
        raise MexcError(f"{method} {path} failed after {self.retries} tries: {last}")

    # ───────────── public ─────────────
    def klines(self, symbol: str, interval: str, start: int, end: int) -> dict:
        """Raw kline arrays for [start, end] (epoch seconds). Open-time stamped."""
        d = self._request("GET", f"/api/v1/contract/kline/{symbol}",
                          {"interval": interval, "start": start, "end": end})
        return d or {"time": []}

    def tickers(self) -> list[dict]:
        d = self._request("GET", "/api/v1/contract/ticker")
        return d if isinstance(d, list) else [d]

    def contract_detail(self, symbol: str) -> dict:
        return self._request("GET", "/api/v1/contract/detail", {"symbol": symbol})

    def funding_history(self, symbol: str, max_pages: int = 200) -> list[tuple[int, float]]:
        """[(settle_time_sec, rate_fraction)] ascending."""
        out, page = [], 1
        while page <= max_pages:
            d = self._request("GET", "/api/v1/contract/funding_rate/history",
                              {"symbol": symbol, "page_num": page, "page_size": 100})
            rows = (d or {}).get("resultList") or []
            out += [(int(r["settleTime"]) // 1000, float(r["fundingRate"])) for r in rows]
            if page >= int((d or {}).get("totalPage", 0) or 0) or not rows:
                break
            page += 1
        return sorted(set(out))

    # ───────────── private ─────────────
    def open_positions(self, symbol: str | None = None) -> list[dict]:
        return self._request("GET", "/api/v1/private/position/open_positions",
                             {"symbol": symbol}, private=True) or []

    def submit_market_order(self, symbol: str, side: int, vol: int, leverage: int,
                            margin_type: str, last_price: float, ext_id: str) -> dict:
        body = {"symbol": symbol, "price": last_price, "vol": vol, "leverage": leverage,
                "side": side, "type": 5, "openType": 1 if margin_type == "isolated" else 2,
                "externalOid": ext_id}
        return self._request("POST", "/api/v1/private/order/submit", body=body, private=True)

    def place_stop(self, position_id: int, vol: int, stop_price: float) -> dict:
        body = {"positionId": position_id, "vol": vol, "stopLossPrice": stop_price,
                "lossTrend": 1, "profitLossVolType": "SAME", "volType": 2}
        return self._request("POST", "/api/v1/private/stoporder/place", body=body, private=True)

    def change_stop(self, stop_order_id: int, stop_price: float) -> dict:
        body = {"stopPlanOrderId": stop_order_id, "stopLossPrice": stop_price, "lossTrend": 1}
        return self._request("POST", "/api/v1/private/stoporder/change_price", body=body, private=True)

    def open_stop_orders(self, symbol: str | None = None) -> list[dict]:
        d = self._request("GET", "/api/v1/private/stoporder/list/orders",
                          {"symbol": symbol, "is_finished": 0, "page_num": 1, "page_size": 100},
                          private=True)
        return (d or {}).get("resultList", []) if isinstance(d, dict) else (d or [])

    def cancel_stop(self, stop_order_id: int) -> dict:
        return self._request("POST", "/api/v1/private/stoporder/cancel",
                             body=[{"stopPlanOrderId": stop_order_id}], private=True)
