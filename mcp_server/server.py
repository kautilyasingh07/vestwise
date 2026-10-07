"""Vestwise MCP server (spec FR-20): the agent's tools over the Model Context Protocol.

Run (stdio):  VESTWISE_USER_ID=u_priya python mcp_server/server.py
Inspect:      npx @modelcontextprotocol/inspector -e VESTWISE_USER_ID=u_priya .venv/bin/python mcp_server/server.py

Exposes get_vesting_status, get_grants and search_policy for ONE user: the one
named by VESTWISE_USER_ID, resolved once at startup with `load_context` (fails
closed). The tools come from `build_tools(ctx)`, the same factory the chat agent
uses, so every access rule applies unchanged: tools are closures over ctx, take
no identity arguments, employees can't name another stakeholder, and
search_policy runs with that user's access filter.

Every tool call writes an audit record (kind "mcp": user, tool, arguments as
sent, outcome, latency, chunk ids) before its result is returned. Fail closed,
as /chat: if the audit write fails, the client gets an error, not the data.

stdout is the protocol channel in stdio mode: never print() here; log to stderr.
Uses mcp 2.x, where FastMCP was renamed MCPServer (same decorator-style API).
"""

import inspect
import json
import logging
import sys
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# `python mcp_server/server.py` puts mcp_server/ (not the repo root) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.tools import BaseTool  # noqa: E402
from mcp.server.mcpserver import Context, MCPServer  # noqa: E402
from mcp.types import CallToolResult, TextContent, ToolAnnotations  # noqa: E402

from app.audit import McpOutcome, write_mcp_audit  # noqa: E402
from app.config import settings  # noqa: E402
from app.context import RequestContext, load_context  # noqa: E402
from app.tools.factory import NO_RESULTS, build_tools  # noqa: E402

log = logging.getLogger("vestwise.mcp")

EXPOSED = ("get_vesting_status", "get_grants", "search_policy")  # FR-20; no cap table or dilution
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True,
                            open_world_hint=False)
AUDIT_FAILED = "Audit log unavailable, so the result was withheld. Please try again later."
MAX_ERROR_CHARS = 500

AuditWriter = Callable[..., str]  # write_mcp_audit(ctx, tool_name, arguments, latency_ms, **kwargs) -> id


@dataclass
class CallTrace:
    """What one tool call saw, collected while it runs (for the audit record)."""

    chunk_ids: list[str] = field(default_factory=list)


# The current call's trace. The tool function may run in a worker thread with a
# copy of this context; the copy still points at the same CallTrace object.
CURRENT_CALL: ContextVar[CallTrace | None] = ContextVar("vestwise_mcp_call", default=None)


def result_text(result: Any) -> str:
    """The first text block of a tool result ("" if there is none)."""
    content = getattr(result, "content", None) or []
    return next((c.text for c in content if isinstance(c, TextContent)), "")


def describe_error(exc: BaseException) -> str:
    """Error text for the audit record, including the cause the SDK wraps.

    The SDK raises UnexpectedToolError("Error executing tool X") from the tool's own
    exception; the client sees only that generic text, the audit record gets the cause too.
    """
    text = f"{type(exc).__name__}: {exc}"
    if exc.__cause__ is not None:
        text += f" (cause: {type(exc.__cause__).__name__}: {exc.__cause__})"
    return text[:MAX_ERROR_CHARS]


def classify_result(result: Any) -> tuple[McpOutcome, str | None]:
    """(outcome, error) for the audit record.

    error     -> the call failed (unknown tool, invalid arguments, exception);
    not_found -> search_policy found nothing above the threshold;
    rejected  -> the tool answered with {"error": ...} (bad date, unknown or ambiguous name);
    ok        -> anything else.
    """
    text = result_text(result)
    if isinstance(result, CallToolResult) and result.is_error:
        return "error", text[:MAX_ERROR_CHARS] or "tool error"
    if text.startswith(NO_RESULTS):
        return "not_found", None
    try:
        data = json.loads(text)
    except ValueError:
        return "ok", None
    if isinstance(data, dict) and "error" in data:
        return "rejected", str(data["error"])[:MAX_ERROR_CHARS]
    return "ok", None


class AuditedServer(MCPServer):
    """MCPServer that audit-logs every tools/call, whatever its outcome.

    `call_tool` is the method the SDK's tools/call handler awaits, for every
    request: unknown tool names and invalid arguments included. Anything raised
    here becomes an `isError` result for the client.
    """

    def __init__(self, ctx: RequestContext, audit: AuditWriter, name: str, **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self.ctx, self.audit = ctx, audit

    async def call_tool(self, name: str, arguments: dict[str, Any], context: Context | None = None) -> Any:
        """Run the tool, write the audit record, then (and only then) return the result."""
        trace, start = CallTrace(), time.monotonic()
        token = CURRENT_CALL.set(trace)
        try:
            result = await super().call_tool(name, arguments, context)
        except Exception as exc:
            error = describe_error(exc)
            try:
                self.write(name, arguments, start, trace, outcome="error", error=error)
            except Exception:  # noqa: BLE001 - the original error is what the client needs to hear about
                log.exception("audit write failed after an MCP tool error")
            raise
        finally:
            CURRENT_CALL.reset(token)

        outcome, error = classify_result(result)
        try:
            self.write(name, arguments, start, trace, outcome=outcome, error=error)
        except Exception:  # noqa: BLE001 - fail closed: no result leaves without an audit record
            log.exception("audit write failed for user %s, tool %s", self.ctx.user_id, name)
            raise RuntimeError(AUDIT_FAILED) from None
        return result

    def write(self, name: str, arguments: dict[str, Any], start: float, trace: CallTrace, *,
              outcome: McpOutcome, error: str | None) -> str:
        """One audit record for this call."""
        latency_ms = round((time.monotonic() - start) * 1000)
        return self.audit(self.ctx, name, arguments, latency_ms, outcome=outcome,
                          chunk_ids=trace.chunk_ids, error=error)


def resolve_context(user_id: str | None) -> RequestContext:
    """Return the server's fixed identity, or raise SystemExit if it's missing or unusable."""
    if not user_id or not user_id.strip():
        raise SystemExit("VESTWISE_USER_ID is not set: refusing to start (the server must know whose "
                         "data it serves).")
    try:
        return load_context(user_id.strip())
    except (LookupError, ValueError) as exc:  # UnknownUserError is a LookupError; bad records are ValueError
        raise SystemExit(f"VESTWISE_USER_ID={user_id!r} is not a usable user: {exc}. Refusing to start.") from exc


def mcp_functions(ctx: RequestContext, run: Callable[[str, dict[str, Any]], str]) -> list[Callable[..., str]]:
    """Plain functions MCP can introspect (it builds each tool's schema from the signature).

    Employees' functions have no stakeholder parameter at all; the admin's
    mirror the admin tools in build_tools (stakeholder_name, resolved within
    the admin's company). Each body only forwards to the build_tools tool.
    """

    def search_policy(query: str) -> str:
        return run("search_policy", {"query": query})

    if ctx.is_admin:

        def get_vesting_status(as_of: str | None = None, stakeholder_name: str | None = None) -> str:
            return run("get_vesting_status", {"as_of": as_of, "stakeholder_name": stakeholder_name})

        def get_grants(stakeholder_name: str | None = None) -> str:
            return run("get_grants", {"stakeholder_name": stakeholder_name})

    else:

        def get_vesting_status(as_of: str | None = None) -> str:  # type: ignore[misc]
            return run("get_vesting_status", {"as_of": as_of})

        def get_grants() -> str:  # type: ignore[misc]
            return run("get_grants", {})

    return [get_vesting_status, get_grants, search_policy]


def build_server(ctx: RequestContext, audit: AuditWriter) -> AuditedServer:
    """An MCP server whose three tools act for `ctx` only, via build_tools(ctx).

    `audit` writes one record per tool call (main() passes write_mcp_audit; tests pass a fake).
    """
    tools: dict[str, BaseTool] = {t.name: t for t in build_tools(ctx) if t.name in EXPOSED}

    def run(name: str, args: dict[str, Any]) -> str:
        log.info("tool call user=%s tool=%s args=%s", ctx.user_id, name, args)
        # Invoked as a ToolCall so we get a ToolMessage: .content is the text for the
        # client, .artifact is search_policy's chunks (their ids go into the audit record).
        message = tools[name].invoke({"name": name, "args": args, "id": f"mcp-{name}", "type": "tool_call"})
        trace = CURRENT_CALL.get()
        if trace is not None and isinstance(message.artifact, list):
            trace.chunk_ids.extend(c.chunk_id for c in message.artifact)
        return message.content

    server = AuditedServer(
        ctx,
        audit,
        "vestwise",
        instructions=(f"ESOP tools for {ctx.name} ({ctx.role}) at company {ctx.company_id}. "
                      "Cite policy facts with the [Title, p. N] tags search_policy returns; "
                      "numbers come only from tool results."),
    )
    for fn in mcp_functions(ctx, run):
        tool = tools[fn.__name__]
        params, expected = set(inspect.signature(fn).parameters), set(tool.args)
        if params != expected:  # build_tools changed and this file didn't: fail at startup, not mid-call
            raise RuntimeError(f"{fn.__name__}: MCP params {params} != build_tools params {expected}")
        server.add_tool(fn, name=tool.name, description=tool.description, annotations=READ_ONLY)
    return server


def main() -> None:
    """Resolve the fixed user, build the server, serve over stdio."""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ctx = resolve_context(settings.vestwise_user_id)
    log.info("serving %s (%s, %s) over stdio", ctx.user_id, ctx.role, ctx.company_id)
    build_server(ctx, write_mcp_audit).run("stdio")


if __name__ == "__main__":
    main()
