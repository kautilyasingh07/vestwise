"""HTTP client for the Vestwise API: the only way the UI gets data (spec §6, FR-19).

No Mongo, no LLM, no app internals: just requests to API_URL with the
X-User-Id header. Every failure (timeout, connection refused, 401/403/422/5xx,
a non-JSON body) becomes an `ApiResult` with a human-readable `error`, so the
UI can show a message instead of a stack trace.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any

import requests

TIMEOUT_SECONDS = 60  # free-tier LLMs are throttled; Phase 5/6 saw single answers take ~45 s
COMPLIANCE_TIMEOUT_SECONDS = 120
AUDIT_LIMIT = 20


@dataclass(frozen=True)
class ApiResult:
    """Outcome of one API call: `data` when ok, otherwise a friendly `error`."""

    ok: bool
    status: int | None = None
    data: Any = None
    error: str | None = None
    audit_id: str | None = None


def _detail(response: requests.Response) -> str:
    """The API's `detail` field as text (FastAPI 422 details are a list of problems)."""
    try:
        detail = response.json().get("detail", "")
    except (ValueError, AttributeError):
        return ""
    if isinstance(detail, list):
        return "; ".join(f"{'.'.join(str(p) for p in d.get('loc', [])[1:])}: {d.get('msg', '')}" for d in detail)
    return str(detail)


def error_message(status: int, detail: str) -> str:
    """Map an HTTP error status to a message a user can act on."""
    if status == 401:
        return "You're not signed in: the API doesn't recognise this user. Pick a user in the sidebar."
    if status == 403:
        return f"You don't have access to this. {detail}".strip()
    if status == 404:
        return f"Not found. {detail}".strip()
    if status == 422:
        return f"The request wasn't valid: {detail}" if detail else "The request wasn't valid."
    if status >= 500:
        return detail or "The server had a problem. Please try again."
    return f"Unexpected response from the API (HTTP {status}). {detail}".strip()


def call(
    base_url: str,
    method: str,
    path: str,
    user_id: str,
    *,
    json: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    files: dict[str, tuple[str, bytes, str]] | None = None,
    timeout: float = TIMEOUT_SECONDS,
) -> ApiResult:
    """Send one request as `user_id` (X-User-Id header); never raises."""
    url = base_url.rstrip("/") + path
    extra = {"files": files} if files is not None else {}
    try:
        response = requests.request(method, url, headers={"X-User-Id": user_id}, json=json, params=params,
                                    timeout=timeout, **extra)
    except requests.Timeout:
        return ApiResult(ok=False, error=f"No answer within {timeout:.0f} s. Free-tier models are sometimes "
                                         "throttled; please try again in a moment.")
    except requests.ConnectionError:
        return ApiResult(ok=False, error=f"Can't reach the API at {base_url}. Is it running? "
                                         "Start everything with `bash scripts/run_all.sh`.")
    except requests.RequestException as exc:
        return ApiResult(ok=False, error=f"Request failed: {type(exc).__name__}")

    if not response.ok:
        return ApiResult(ok=False, status=response.status_code,
                         error=error_message(response.status_code, _detail(response)))
    try:
        data = response.json()
    except ValueError:
        return ApiResult(ok=False, status=response.status_code, error="The API returned a response that isn't JSON.")
    return ApiResult(ok=True, status=response.status_code, data=data, audit_id=response.headers.get("X-Audit-Id"))


def chat(base_url: str, user_id: str, message: str, history: list[dict[str, str]], as_of: date) -> ApiResult:
    """POST /chat. Example: chat(url, "u_priya", "How many options have I vested?", [], date(2026, 10, 3))."""
    return call(base_url, "POST", "/chat", user_id,
                json={"message": message, "history": history, "as_of": as_of.isoformat()})


def cap_table(base_url: str, user_id: str) -> ApiResult:
    """GET /captable (admin only)."""
    return call(base_url, "GET", "/captable", user_id)


def simulate(base_url: str, user_id: str, new_shares: int, investor_name: str) -> ApiResult:
    """POST /captable/simulate (admin only)."""
    return call(base_url, "POST", "/captable/simulate", user_id,
                json={"new_shares": new_shares, "investor_name": investor_name})


def compliance_check(base_url: str, user_id: str, file_name: str, data: bytes) -> ApiResult:
    """POST /compliance/check with the PDF as multipart (admin only). Up to two LLM calls, hence the longer timeout."""
    return call(base_url, "POST", "/compliance/check", user_id,
                files={"file": (file_name, data, "application/pdf")}, timeout=COMPLIANCE_TIMEOUT_SECONDS)


def audit(base_url: str, user_id: str, limit: int = AUDIT_LIMIT) -> ApiResult:
    """GET /audit (admin only), newest first."""
    return call(base_url, "GET", "/audit", user_id, params={"limit": limit})
