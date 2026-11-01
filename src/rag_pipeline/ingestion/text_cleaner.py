"""Conservative, idempotent text cleaning.

Only invisible noise is changed. Headings, numbers, negation, indentation, and interior
spacing (which can be meaningful in code and tables) are preserved. Headers/footers are
not removed and hyphenation is not repaired: those heuristics can delete real content.
"""

import re

_LINE_ENDINGS = re.compile(r"\r\n?|[\x0b\x0c\x85  ]")
_CONTROL = re.compile(r"[\x00-\x08\x0e-\x1f\x7f﻿]")
_TRAILING_SPACE = re.compile(r"[ \t]+(?=\n)")
_BLANK_RUNS = re.compile(r"\n{3,}")


def clean_text(raw: str) -> str:
    text = _LINE_ENDINGS.sub("\n", raw)
    text = _CONTROL.sub("", text)
    text = _TRAILING_SPACE.sub("", text)
    text = _BLANK_RUNS.sub("\n\n", text)
    return text.strip()
