"""Request context: who is asking (spec §6 request flow, step 2).

Built server-side from the `users` collection. Nothing in it comes from the
LLM or the request body; the tool factory closes over it, so the model can
never choose whose data a tool reads.
"""

from dataclasses import dataclass

from app import db
from app.rag.access import Role

ROLES: tuple[Role, ...] = ("employee", "admin")


class UnknownUserError(LookupError):
    """Raised when a user id is not in the `users` collection."""


@dataclass(frozen=True)
class RequestContext:
    """Identity for one request. Frozen: nothing downstream can change who is asking."""

    user_id: str
    role: Role
    company_id: str
    stakeholder_id: str | None
    name: str

    @property
    def is_admin(self) -> bool:
        """True for company admins (HR / founders)."""
        return self.role == "admin"


def load_context(user_id: str) -> RequestContext:
    """Look up a user and build their context; fails closed on unknown users or bad records.

    Example: load_context("u_priya") ->
        RequestContext(user_id="u_priya", role="employee", company_id="nimbus",
                       stakeholder_id="sh_priya", name="Priya Sharma")
    """
    user = db.users().find_one({"_id": user_id})
    if user is None:
        raise UnknownUserError(f"unknown user {user_id!r}")
    role = user.get("role")
    if role not in ROLES:
        raise ValueError(f"user {user_id!r} has invalid role {role!r}")
    if not user.get("company_id"):
        raise ValueError(f"user {user_id!r} has no company_id")
    if role == "employee" and not user.get("stakeholder_id"):
        raise ValueError(f"employee {user_id!r} has no stakeholder_id")
    return RequestContext(
        user_id=user_id,
        role=role,
        company_id=user["company_id"],
        stakeholder_id=user.get("stakeholder_id"),
        name=user.get("name", user_id),
    )
