"""Per-request LangChain tools, built as closures over the RequestContext (spec §8.6, FR-17).

The security rule: no tool takes company_id, role or stakeholder_id. Those come
from `ctx`, which each tool captures when `build_tools` runs. The LLM only
fills in the parameters a tool declares, so a prompt injection can't widen
access: the parameter it would need doesn't exist. Admin-only tools are simply
not created for employees.

Tool results are JSON strings (dates as ISO), so the model reads exact figures
and never has to compute them (FR-15). search_policy also returns its chunks as
a LangChain *artifact*, which the agent turns into citations without showing
them to the model twice.
"""

import json
from datetime import date
from typing import Any

from langchain_core.tools import BaseTool, tool

from app.config import settings
from app.context import RequestContext
from app.rag.retriever import RetrievedChunk, retrieve
from app.tools import repo
from app.tools.captable import cap_table, simulate_dilution
from app.tools.vesting import compute_vesting

SEARCH_POLICY = "search_policy"
NO_RESULTS = "NO_RESULTS: nothing in the company's documents matches this query."

# Grant fields shown to the model (internal ids like company_id stay out).
GRANT_FIELDS = ("grant_date", "options", "strike_price", "vesting", "exercised", "lapsed",
                "status", "termination_date")


def to_json(obj: Any) -> str:
    """Serialise a tool result; dates become YYYY-MM-DD."""
    return json.dumps(obj, default=lambda v: v.isoformat() if isinstance(v, date) else str(v))


def citation_tag(chunk: RetrievedChunk) -> str:
    """The tag the model must copy into its answer, e.g. "[ESOP Policy, p. 5]"."""
    return f"[{chunk.doc_title}, p. {chunk.page}]"


def format_chunks(chunks: list[RetrievedChunk]) -> str:
    """Chunks as the model sees them: tag and section, then the text."""
    if not chunks:
        return NO_RESULTS
    return "\n\n".join(f"{citation_tag(c)} {c.section}\n{c.text}" for c in chunks)


def parse_as_of(value: str | None, default: date) -> date:
    """Parse a YYYY-MM-DD date from the model, or use the default."""
    return date.fromisoformat(value) if value else default


def resolve_stakeholder(company_id: str, name: str) -> tuple[str, str]:
    """Map a name typed by an admin to (stakeholder_id, full name) within their company.

    Matches the full name or a single word of it, case-insensitively; raises
    LookupError if nothing or more than one stakeholder matches.
    """
    wanted = name.strip().lower()
    names = repo.get_stakeholder_names(company_id)
    exact = [(sid, n) for sid, n in names.items() if n.lower() == wanted]
    matches = exact or [(sid, n) for sid, n in names.items() if wanted in n.lower().split()]
    if len(matches) != 1:
        known = ", ".join(sorted(names.values()))
        problem = "matches several people" if matches else "matches no one"
        raise LookupError(f"'{name}' {problem}. Stakeholders: {known}")
    return matches[0]


def build_tools(ctx: RequestContext, as_of: date | None = None) -> list[BaseTool]:
    """Return the tools this user may call, each closed over `ctx`.

    Employee: search_policy, get_vesting_status(as_of), get_grants().
    Admin:    the same, with an optional stakeholder_name on the grant tools,
              plus get_cap_table() and simulate_dilution(new_shares, investor_name).
    `as_of` fixes "today" for tests and eval; otherwise it's the real date at call time.
    """

    def today() -> date:
        return as_of or date.today()

    def vesting_for(stakeholder_id: str | None, holder: str, when: date) -> str:
        grants = repo.get_grants(ctx.company_id, stakeholder_id) if stakeholder_id else []
        results = [compute_vesting(g, when, settings.exercise_window_days) for g in grants]
        return to_json({"holder": holder, "as_of": when, "grants": results} if results
                       else {"holder": holder, "as_of": when, "grants": [], "note": "No option grants."})

    def grants_for(stakeholder_id: str | None, holder: str) -> str:
        grants = repo.get_grants(ctx.company_id, stakeholder_id) if stakeholder_id else []
        rows = [{"grant_id": g["_id"], **{k: g.get(k) for k in GRANT_FIELDS}} for g in grants]
        return to_json({"holder": holder, "grants": rows} if rows
                       else {"holder": holder, "grants": [], "note": "No option grants."})

    @tool(SEARCH_POLICY, response_format="content_and_artifact")
    def search_policy(query: str) -> tuple[str, list[RetrievedChunk]]:
        """Search the company's ESOP documents (ESOP policy, board resolution, and the user's own grant letter).

        Use for any policy question: vesting rules, cliff, exercise window, leaving the company,
        good/bad leaver, acquisition, tax, transfers, eligibility.
        Write `query` in the policy's own terms, not the user's words: drop the company
        name and greetings, and use the policy's vocabulary, e.g. "acquired" -> "change of
        control acquisition", "quit"/"resign" -> "termination of employment, unvested options lapse",
        "how long to buy after leaving" -> "exercise window after leaving".
        Each result starts with a tag like [ESOP Policy, p. 5]; cite facts with that exact tag.
        Returns NO_RESULTS if nothing relevant is found.

        Example: user asks "If Nimbus lets me go, how long can I still buy my shares?"
        -> search_policy(query="exercise window after leaving, termination of employment")
        """
        chunks = retrieve(query, ctx.company_id, ctx.role, ctx.stakeholder_id)
        return format_chunks(chunks), chunks

    if ctx.is_admin:

        @tool("get_vesting_status")
        def get_vesting_status_admin(as_of: str | None = None, stakeholder_name: str | None = None) -> str:
            """Vesting status of a stakeholder's option grants on a date: granted, vested, unvested,
            exercised, exercisable, lapsed, next vest date and options, exercise deadline if they left.

            as_of: YYYY-MM-DD; omit for today. stakeholder_name: e.g. "Kiran" or "Priya Sharma";
            omit for the signed-in admin.
            Example: "How many options has Kiran exercised?" -> get_vesting_status(stakeholder_name="Kiran")
            """
            try:
                when = parse_as_of(as_of, today())
                if stakeholder_name:
                    sid, holder = resolve_stakeholder(ctx.company_id, stakeholder_name)
                else:
                    sid, holder = ctx.stakeholder_id, ctx.name
            except (ValueError, LookupError) as exc:
                return to_json({"error": str(exc)})
            return vesting_for(sid, holder, when)

        @tool("get_grants")
        def get_grants_admin(stakeholder_name: str | None = None) -> str:
            """Option grants of a stakeholder: grant date, number of options, strike price,
            vesting schedule, exercised, lapsed, status, termination date.

            stakeholder_name: e.g. "Rahul"; omit for the signed-in admin.
            Example: "What is Rahul's strike price?" -> get_grants(stakeholder_name="Rahul")
            """
            try:
                sid, holder = (resolve_stakeholder(ctx.company_id, stakeholder_name) if stakeholder_name
                               else (ctx.stakeholder_id, ctx.name))
            except LookupError as exc:
                return to_json({"error": str(exc)})
            return grants_for(sid, holder)

        @tool("get_cap_table")
        def get_cap_table() -> str:
            """The company's cap table: each holder's shares and outstanding options, issued % and
            fully diluted %, plus the unallocated ESOP pool and totals.

            Example: "Who owns what percentage of the company?" -> get_cap_table()
            """
            names = repo.get_stakeholder_names(ctx.company_id)
            table = cap_table(repo.get_holdings(ctx.company_id), repo.get_grants(ctx.company_id),
                              repo.get_pool_size(ctx.company_id), names)
            return to_json(table)

        @tool("simulate_dilution")
        def simulate_dilution_tool(new_shares: int, investor_name: str) -> str:
            """Before/after ownership for every holder (issued and fully diluted %) if the company
            issues `new_shares` new shares to a new investor called `investor_name`.

            Example: "What if we issue 2,000,000 shares to Horizon Capital?"
            -> simulate_dilution(new_shares=2000000, investor_name="Horizon Capital")
            """
            try:
                result = simulate_dilution(
                    repo.get_holdings(ctx.company_id), repo.get_grants(ctx.company_id),
                    repo.get_pool_size(ctx.company_id), new_shares, investor_name,
                    repo.get_stakeholder_names(ctx.company_id))
            except ValueError as exc:
                return to_json({"error": str(exc)})
            return to_json(result)

        return [search_policy, get_vesting_status_admin, get_grants_admin, get_cap_table, simulate_dilution_tool]

    @tool("get_vesting_status")
    def get_vesting_status(as_of: str | None = None) -> str:
        """The signed-in user's own vesting status on a date: granted, vested, unvested, exercised,
        exercisable, lapsed, next vest date and options, exercise deadline if they have left.

        as_of: YYYY-MM-DD; omit for today. Use a future date for "next month" or "in a year".
        Example: "How many options will I have vested by 2027-01-01?" -> get_vesting_status(as_of="2027-01-01")
        """
        try:
            when = parse_as_of(as_of, today())
        except ValueError as exc:
            return to_json({"error": str(exc)})
        return vesting_for(ctx.stakeholder_id, ctx.name, when)

    @tool("get_grants")
    def get_grants() -> str:
        """The signed-in user's own option grants: grant date, number of options, strike price,
        vesting schedule, exercised, lapsed, status.

        Example: "What is my strike price?" -> get_grants()
        """
        return grants_for(ctx.stakeholder_id, ctx.name)

    return [search_policy, get_vesting_status, get_grants]
