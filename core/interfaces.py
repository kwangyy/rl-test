"""The two tool signatures frozen in week 1 (docs/plan.md and docs/trace-format.md).

Frozen 2026-10-11. The loop (AG-2) calls these; AG-1 implements `search`,
RL-1's trained model sits behind `verify`. Implementations live elsewhere and
only have to match these shapes. Field names follow docs/trace-format.md,
which logs both calls as steps.

Fields may be added after freezing, never renamed or removed.
"""

from typing import Protocol, TypedDict

Label = str  # "SUPPORTED" | "REFUTED" | "NOT_ENOUGH_INFO", untrained baseline's spelling
EvidenceId = str  # "<page>:<line>", e.g. "Nikolaj_Coster-Waldau:7"; never a prompt-local "E3"


class Hit(TypedDict):
    page: str  # FEVER page id, underscores for spaces
    line: int  # FEVER line number on that page, as in pages.sqlite
    text: str  # the full sentence
    score: float  # retriever score; higher is better, comparable only within one call


class Verdict(TypedDict):
    verdict: Label
    evidence: list[EvidenceId]  # at most 5, a subset of the evidence passed in; [] for NOT_ENOUGH_INFO
    confidence: float  # P(verdict is correct): the verdict-token probability, not the written number
    token_prob: dict[Label, float]  # distribution over the three labels, sums to 1


class Search(Protocol):
    def __call__(self, query: str, k: int = 5) -> list[Hit]:
        """Up to k hits for the query, best first."""


class Verify(Protocol):
    def __call__(self, claim: str, evidence: list[EvidenceId]) -> Verdict:
        """Judge the claim against exactly these sentences.

        The implementation looks up the text of each id and builds its own
        prompt (untrained baseline's E1..En numbering is internal to it); the caller only
        ever sees global "page:line" ids.
        """


class Decompose(Protocol):
    def __call__(self, claim: str) -> list[str]:
        """Sub-claims that are all true exactly when the claim is true.

        Added 2026-10-15 for the agent loop's `decompose` action (AG-2). A claim
        that is already simple comes back as [claim], never as [].
        """
