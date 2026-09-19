"""A 12-claim stand-in for FEVER, so the scaffold runs before any dataset
is downloaded.

The point of the toy corpus is not realism -- it is to pin down the
*interface* the real loader must satisfy (claim, gold label, gold evidence
sentence ids, a retrievable sentence pool), and to make the env's shape
argue-able before committing to a 185K-claim pipeline.

Deliberately included, because each one exercises a controller action the
plan claims is needed:

- compound claims (`decompose`): two independently checkable halves, one
  true and one false, so a model that verifies only the first half is wrong.
- claims whose evidence is genuinely absent (`abstain`): gold label NEI,
  and no sentence in the pool settles them.
- claims whose *phrasing* leaks the label but whose evidence does not
  support it (`evidence_f1` pressure): the known FEVER shortcut, where a
  model can hit the right verdict for the wrong reason.
"""

from __future__ import annotations

SUPPORTED, REFUTED, NEI = "SUPPORTED", "REFUTED", "NEI"

# sentence_id -> sentence text
SENTENCES = {
    "s0": "The Eiffel Tower is located in Paris, France.",
    "s1": "The Eiffel Tower was completed in 1889.",
    "s2": "Paris is the capital of France.",
    "s3": "Mount Everest has an elevation of 8,849 metres.",
    "s4": "Mount Everest lies on the border between Nepal and China.",
    "s5": "The Amazon River is located in South America.",
    "s6": "The Nile is generally regarded as the longest river in the world.",
    "s7": "Python was first released in 1991 by Guido van Rossum.",
    "s8": "Python is a dynamically typed programming language.",
    "s9": "The Great Wall of China is not visible to the naked eye from space.",
    "s10": "Insulin was first isolated at the University of Toronto in 1921.",
    "s11": "Insulin is used in the treatment of diabetes.",
    "s12": "The 2020 Summer Olympics were held in Tokyo in 2021.",
    "s13": "Venus is the hottest planet in the Solar System.",
    "s14": "Mercury is the closest planet to the Sun.",
}

# claim, gold label, gold evidence ids, whether it is compound
CLAIMS = [
    ("The Eiffel Tower is in Paris.", SUPPORTED, ["s0"], False),
    ("The Eiffel Tower was completed in 1912.", REFUTED, ["s1"], False),
    ("Mount Everest is over 8000 metres tall.", SUPPORTED, ["s3"], False),
    ("The Amazon is the longest river in the world.", REFUTED, ["s6"], False),
    ("Python was released in 1991 and is statically typed.", REFUTED, ["s7", "s8"], True),
    ("Insulin was isolated in Toronto and treats diabetes.", SUPPORTED, ["s10", "s11"], True),
    ("The Great Wall of China is visible from space with the naked eye.", REFUTED, ["s9"], False),
    ("The 2020 Olympics were held in Tokyo.", SUPPORTED, ["s12"], False),
    ("Mercury is the hottest planet in the Solar System.", REFUTED, ["s13", "s14"], False),
    # no sentence in the pool settles these -- the abstain cases
    ("The Eiffel Tower receives seven million visitors each year.", NEI, [], False),
    ("Guido van Rossum named Python after a snake.", NEI, [], False),
    ("Everest summit permits cost more in 2026 than in 2025.", NEI, [], False),
]

_STOP = {"the", "is", "in", "a", "of", "was", "and", "are", "to", "it", "from", "with", "than"}


def tokens(text: str) -> set[str]:
    return {w.strip(".,").lower() for w in text.split()} - _STOP


def retrieve(query: str, k: int, exclude: set[str] | None = None) -> list[tuple[str, float]]:
    """Token-overlap stand-in for BM25. Returns (sentence_id, score), best first.

    Swap this for a real BM25 + dense retriever before any conclusion about
    the controller is trustworthy -- the plan's own R1 risk is that a weak
    retriever caps everything downstream, so this must be replaced, not tuned.
    """
    exclude = exclude or set()
    q = tokens(query)
    scored = []
    for sid, sent in SENTENCES.items():
        if sid in exclude:
            continue
        s = tokens(sent)
        overlap = len(q & s)
        if overlap:
            scored.append((sid, overlap / len(q | s)))
    scored.sort(key=lambda x: -x[1])
    return scored[:k]


def decompose(claim: str) -> list[str]:
    """Split a compound claim on 'and'. A placeholder for the Graph-of-Thought
    step -- real decomposition needs a model call, and its error rate is
    tracked as its own metric rather than folded into final accuracy.
    """
    parts = [p.strip() for p in claim.split(" and ") if p.strip()]
    return parts if len(parts) > 1 else [claim]
