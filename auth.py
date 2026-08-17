"""Single shared email/password gate for Streamlit Cloud. No RBAC."""

from __future__ import annotations

import hmac
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")


def _from_secrets(name: str) -> str:
    try:
        value = st.secrets[name]
    except Exception:
        return ""
    if value is None:
        return ""
    return str(value).strip()


def credential(name: str) -> str:
    return _from_secrets(name) or (os.environ.get(name) or "").strip()


def auth_configured() -> tuple[str, str]:
    return credential("DASHBOARD_EMAIL"), credential("DASHBOARD_PASSWORD")


def require_login() -> None:
    """
    Block the dashboard until the shared email/password match.
    If both secrets are unset, skip (local/dev). If only one is set, stop.
    """
    email, password = auth_configured()
    if not email and not password:
        return
    if not email or not password:
        st.error(
            "Auth is misconfigured. Set both DASHBOARD_EMAIL and DASHBOARD_PASSWORD "
            "in Streamlit secrets or `.env`."
        )
        st.stop()

    if st.session_state.get("authenticated"):
        return

    st.title("AI Test Plan Comparator")
    st.subheader("Sign in")
    st.caption("This app is locked. Use the shared email and password.")
    with st.form("dashboard_login"):
        entered_email = st.text_input("Email")
        entered_password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        email_ok = hmac.compare_digest(entered_email.strip().lower(), email.lower())
        password_ok = hmac.compare_digest(entered_password.strip(), password)
        if email_ok and password_ok:
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Invalid email or password.")
    st.stop()


def render_logout() -> None:
    email, password = auth_configured()
    if not (email and password and st.session_state.get("authenticated")):
        return
    st.sidebar.caption(f"Signed in as {email}")
    if st.sidebar.button("Log out", type="secondary"):
        st.session_state["authenticated"] = False
        st.rerun()
