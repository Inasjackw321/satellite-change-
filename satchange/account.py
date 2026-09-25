"""Google Earth Engine sign-in, sign-out and project discovery."""

import json
import os
import socket
import threading

import ee
import requests

REVOKE_URL = "https://oauth2.googleapis.com/revoke"
PROJECTS_URL = "https://cloudresourcemanager.googleapis.com/v1/projects"
REGISTER_URL = "https://code.earthengine.google.com/register"


def credentials_path():
    return ee.oauth.get_credentials_path()


def is_signed_in():
    return os.path.exists(credentials_path())


class NotSignedIn(Exception):
    pass


def connect(project):
    """Initialise Earth Engine for ``project``."""
    if not is_signed_in():
        raise NotSignedIn("Not signed in to Google Earth Engine.")
    ee.Initialize(project=project)


def start_sign_in():
    """Open Google's sign-in page and wait for it in a background thread, so
    the app stays responsive. is_signed_in() turns true once it completes."""
    with socket.socket() as s:  # a fresh port, in case an earlier attempt is still waiting
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    thread = threading.Thread(
        target=ee.Authenticate, kwargs={"auth_mode": f"localhost:{port}", "force": True}, daemon=True,
    )
    thread.start()
    return thread


def sign_out():
    """Revoke the stored Google token and delete it from this computer.

    Earth Engine keeps one sign-in per computer, so this also signs out other
    Earth Engine tools here (e.g. the earthengine command line)."""
    path = credentials_path()
    try:
        with open(path) as f:
            token = json.load(f).get("refresh_token")
    except (OSError, ValueError):
        token = None
    if token:
        try:  # best effort: the local sign-in is removed either way
            requests.post(REVOKE_URL, params={"token": token}, timeout=10)
        except requests.RequestException:
            pass
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    ee.Reset()


def list_projects():
    """Google Cloud project IDs this account can use, or [] if they can't be listed."""
    from google.auth.transport.requests import AuthorizedSession
    from google.oauth2.credentials import Credentials

    try:
        session = AuthorizedSession(Credentials(None, **ee.oauth.get_credentials_arguments()))
        r = session.get(PROJECTS_URL, params={"filter": "lifecycleState:ACTIVE", "pageSize": 500},
                        timeout=15)
        r.raise_for_status()
        return sorted(p["projectId"] for p in r.json().get("projects", []))
    except Exception:
        return []


def explain_error(error, project):
    """Turn an Earth Engine / Google error into a short, actionable message."""
    text = str(error)
    low = text.lower()
    enable = f"https://console.cloud.google.com/apis/library/earthengine.googleapis.com?project={project}"
    if "invalid_grant" in low or "expired or revoked" in low or "reauthenticate" in low:
        return "Your Google sign-in has expired. Sign out and sign in again."
    if "not registered" in low or "not signed up" in low or ("register" in low and "earth engine" in low):
        return (f"Project **{project}** isn't registered for Earth Engine yet. "
                f"[Register it here]({REGISTER_URL}) (free for noncommercial use), then try again.")
    if "has not been used in project" in low or "service_disabled" in low or "is disabled" in low:
        return (f"The Earth Engine API is switched off for project **{project}**. "
                f"[Turn it on here]({enable}), wait a minute, then try again.")
    if "permission" in low and ("denied" in low or "does not have" in low):
        return (f"Your Google account doesn't have access to project **{project}**. "
                "Check the project name, or pick another project.")
    if "not found" in low and "project" in low:
        return f"Project **{project}** doesn't exist. Check the spelling of the project ID."
    if "too many" in low or "quota" in low or "rate limit" in low:
        return "Earth Engine is busy or your usage limit was reached. Wait a minute and try again."
    return text
