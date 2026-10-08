"""行情 HTTP 客户端：自适应代理模式（代理优先 → 直连回退）。

不同网络环境下"能否直连行情站点"差别很大：

* 有本地代理（Clash/V2Ray 等，常见端口 7897）时——**走代理能通，直连反而被拒**；
* 纯国内网络时——直连即可，代理反而会失败。

因此这里默认 ``auto``：先用系统代理试，失败再直连；两者都失败才报错。
可用环境变量 ``ASHARE_PROXY_MODE`` 固定为 ``proxy`` / ``direct`` / ``auto``。
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

__all__ = ["MarketDataClient", "proxy_mode", "network_diagnose"]

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120 Safari/537.36")


def proxy_mode() -> str:
    return str(os.environ.get("ASHARE_PROXY_MODE", "auto")).strip().lower()


class MarketDataClient:
    """带代理自适应的 HTTP 客户端（线程内各自持有 session）。"""

    def __init__(self, timeout: float = 15.0, retry: int = 2, mode: Optional[str] = None):
        self.timeout = float(timeout)
        self.retry = int(retry)
        self.mode = (mode or proxy_mode()).lower()
        self._sessions: Dict[bool, Any] = {}
        self.last_error = ""

    # ---------------- session ----------------
    def session(self, use_proxy: bool):
        if use_proxy not in self._sessions:
            import requests
            s = requests.Session()
            s.trust_env = bool(use_proxy)      # True=走系统代理，False=强制直连
            s.headers.update({"User-Agent": _UA, "Referer": "https://quote.eastmoney.com/"})
            self._sessions[use_proxy] = s
        return self._sessions[use_proxy]

    def _modes(self):
        if self.mode == "proxy":
            return [True]
        if self.mode == "direct":
            return [False]
        return [True, False]                   # auto：代理优先，失败回退直连

    # ---------------- 请求 ----------------
    def get_json(self, url: str, params: Optional[dict] = None,
                 timeout: Optional[float] = None, headers: Optional[dict] = None) -> Any:
        """GET 并解析 JSON；按模式自动切换代理/直连。失败抛 RuntimeError。"""
        last = ""
        for use_proxy in self._modes():
            sess = self.session(use_proxy)
            for attempt in range(self.retry + 1):
                try:
                    r = sess.get(url, params=params, timeout=timeout or self.timeout,
                                 headers=headers)
                    if r.status_code == 200:
                        return r.json()
                    last = f"HTTP {r.status_code}"
                except Exception as exc:      # 网络/代理异常
                    last = f"{type(exc).__name__}: {str(exc)[:120]}"
                time.sleep(0.35 * (attempt + 1))
        self.last_error = last
        raise RuntimeError(f"行情请求失败（proxy_mode={self.mode}）：{last}")

    def get_text(self, url: str, params: Optional[dict] = None,
                 timeout: Optional[float] = None, encoding: str = "utf-8",
                 headers: Optional[dict] = None) -> str:
        """GET 并返回文本；``headers`` 可覆盖 Referer 等（新浪接口必须带新浪 Referer）。"""
        last = ""
        for use_proxy in self._modes():
            sess = self.session(use_proxy)
            for attempt in range(self.retry + 1):
                try:
                    r = sess.get(url, params=params, timeout=timeout or self.timeout,
                                 headers=headers)
                    if r.status_code == 200:
                        r.encoding = encoding
                        return r.text
                    last = f"HTTP {r.status_code}"
                except Exception as exc:
                    last = f"{type(exc).__name__}: {str(exc)[:120]}"
                time.sleep(0.35 * (attempt + 1))
        self.last_error = last
        raise RuntimeError(f"行情请求失败：{last}")


def network_diagnose(url: str = "https://push2.eastmoney.com/api/qt/ulist.np/get",
                     params: Optional[dict] = None) -> Dict[str, Any]:
    """诊断当前网络：分别测试"走代理"和"直连"是否可用。"""
    import requests
    params = params or {"fltt": "2", "invt": "2", "fields": "f12,f14,f2",
                        "secids": "1.600519"}
    out: Dict[str, Any] = {"env_proxies": {k: v for k, v in os.environ.items()
                                           if "proxy" in k.lower()}, "mode": proxy_mode()}
    try:
        out["system_proxies"] = requests.utils.getproxies()
    except Exception as exc:
        out["system_proxies"] = f"读取失败：{exc}"
    for label, use_proxy in (("代理", True), ("直连", False)):
        s = requests.Session()
        s.trust_env = bool(use_proxy)
        s.headers.update({"User-Agent": _UA})
        t0 = time.time()
        try:
            r = s.get(url, params=params, timeout=8)
            ok = r.status_code == 200 and '"data"' in r.text
            out[label] = {"可用": ok, "HTTP": r.status_code,
                          "耗时ms": int((time.time() - t0) * 1000),
                          "返回": r.text[:60]}
        except Exception as exc:
            out[label] = {"可用": False, "错误": f"{type(exc).__name__}: {str(exc)[:80]}"}
    return out


