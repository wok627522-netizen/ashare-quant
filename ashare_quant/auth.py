"""访问密码保护（登录门禁）。

用法（放在 app.py 最前面，任何数据加载之前）::

    from ashare_quant.auth import require_password, logout_button
    if not require_password():
        st.stop()          # 未登录则不渲染任何内容（防止数据泄露）
    ...
    logout_button()        # 侧边栏的「退出登录」

密码来源（优先级从高到低）：

1. 环境变量 ``ASHARE_PASSWORD``
2. 项目根目录 ``config/auth.json`` 的 ``{"password": "..."}``
3. 默认密码 ``123789``

安全说明：这是**单密码门禁**，适合自己/小团队使用。若要放到公网，请务必：
* 开启 HTTPS（Caddy/Nginx 反代 + 证书）；
* 用强密码（改 ``config/auth.json`` 或环境变量）；
* 最好再加 IP 白名单或 VPN。详见 docs/部署成网站.md。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import streamlit as st

__all__ = ["DEFAULT_PASSWORD", "get_password", "require_password", "logout_button",
           "is_authenticated", "set_password"]

DEFAULT_PASSWORD = "123789"
_MAX_ATTEMPTS = 5
_LOCK_SECONDS = 60


def config_path() -> Path:
    from .config import ROOT
    return Path(ROOT) / "config" / "auth.json"


def get_password() -> str:
    """读取当前生效的密码。

    优先级：Streamlit secrets（云端） > 环境变量 > config/auth.json > 内置默认值。
    """
    # 1) Streamlit Cloud 的 Secrets（推荐云端使用，不会进代码仓库）
    try:
        import streamlit as st
        if "PASSWORD" in st.secrets:
            pw = str(st.secrets["PASSWORD"]).strip()
            if pw:
                return pw
        if "ASHARE_PASSWORD" in st.secrets:
            pw = str(st.secrets["ASHARE_PASSWORD"]).strip()
            if pw:
                return pw
    except Exception:
        pass
    # 2) 环境变量
    env = os.environ.get("ASHARE_PASSWORD")
    if env:
        return str(env)
    p = config_path()
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            pw = str(data.get("password") or "").strip()
            if pw:
                return pw
        except Exception:
            pass
    return DEFAULT_PASSWORD


def set_password(new_password: str) -> Path:
    """写入 config/auth.json（供界面/命令行修改密码）。"""
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"password": str(new_password)}, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return p


def is_authenticated() -> bool:
    return bool(st.session_state.get("_authed", False))


def _css() -> None:
    st.markdown("""
<style>
  .login-wrap {max-width: 460px; margin: 6vh auto 0 auto; padding: 34px 38px;
               background: #fff; border: 1px solid #e6e9ef; border-radius: 18px;
               box-shadow: 0 8px 30px rgba(16,24,40,.08);}
  .login-title {font-size: 26px; font-weight: 700; margin-bottom: 6px; color: #1f2430;}
  .login-sub {color: #667085; font-size: 13px; margin-bottom: 18px;}
  .login-foot {color: #98a2b3; font-size: 12px; margin-top: 14px; text-align: center;}
</style>""", unsafe_allow_html=True)


def require_password(title: str = "A 股量化研究平台",
                     subtitle: str = "请输入访问密码后继续") -> bool:
    """未登录则渲染登录页并返回 False；已登录返回 True。"""
    if is_authenticated():
        return True
    _css()
    st.markdown('<div class="login-wrap">', unsafe_allow_html=True)
    st.markdown(f'<div class="login-title">📈 {title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="login-sub">{subtitle}</div>', unsafe_allow_html=True)

    lock_until = float(st.session_state.get("_lock_until", 0) or 0)
    now = time.time()
    if lock_until > now:
        st.error(f"尝试次数过多，请等待 {int(lock_until - now)} 秒后再试")
        st.markdown('<div class="login-foot">A 股量化研究平台 · 仅供研究使用</div>',
                    unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return False

    with st.form("login_form", clear_on_submit=False):
        pw = st.text_input("访问密码", type="password", placeholder="请输入密码",
                           label_visibility="collapsed")
        ok = st.form_submit_button("🔓 进入系统", type="primary", width="stretch")
    if ok:
        if str(pw) == get_password():
            st.session_state["_authed"] = True
            st.session_state["_fail_count"] = 0
            st.session_state["_login_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            st.rerun()
        else:
            fails = int(st.session_state.get("_fail_count", 0)) + 1
            st.session_state["_fail_count"] = fails
            if fails >= _MAX_ATTEMPTS:
                st.session_state["_lock_until"] = time.time() + _LOCK_SECONDS
                st.session_state["_fail_count"] = 0
                st.error(f"密码错误 {fails} 次，已锁定 {_LOCK_SECONDS} 秒")
            else:
                st.error(f"密码错误（还可尝试 {_MAX_ATTEMPTS - fails} 次）")
    st.markdown('<div class="login-foot">A 股量化研究平台 · 仅供研究使用，不构成投资建议</div>',
                unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    return False


def logout_button(label: str = "🚪 退出登录") -> None:
    """侧边栏退出按钮。"""
    if is_authenticated() and st.sidebar.button(label, key="logout_btn", width="stretch"):
        st.session_state["_authed"] = False
        st.session_state.pop("_login_time", None)
        st.rerun()
