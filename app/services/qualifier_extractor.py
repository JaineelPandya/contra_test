"""Deterministic qualifier extraction for regulatory facts."""
import re
from typing import Any, Dict, List, Optional


def extract_qualifier_spans(text: str) -> Dict[str, Any]:
    """Extract condition, exception, threshold, time, and circular reference spans deterministically.

    Condition signals:
      - provided further that, provided that, on the condition that, subject to, only if, if, where, when
    Exception signals:
      - shall not apply to, does not apply to, except where, except, notwithstanding, excluding, unless
    Threshold signals:
      - monetary amounts, percentages, total annual revenue, net worth
    Time signals:
      - working days, calendar days, deadlines, frequencies
    Source circular signals:
      - SEBI/HO/..., CIR/..., Circular no. ...
    """
    if not text or not text.strip():
        return {
            "conditions": [],
            "exceptions": [],
            "threshold": None,
            "time": None,
            "source_circular": None,
        }

    all_signals = [
        # Exception patterns first (longest / most specific first)
        (r"\bshall\s+not\s+apply\s+to\b", "exception"),
        (r"\bdoes\s+not\s+apply\s+to\b", "exception"),
        (r"\bexcept\s+where\b", "exception"),
        (r"\bexcept\b", "exception"),
        (r"\bnotwithstanding\b", "exception"),
        (r"\bexcluding\b", "exception"),
        (r"\bunless\b", "exception"),
        # Condition patterns (longest / most specific first)
        (r"\bprovided\s+further\s+that\b", "condition"),
        (r"\bprovided\s+that\b", "condition"),
        (r"\bon\s+the\s+condition\s+that\b", "condition"),
        (r"\bsubject\s+to\b", "condition"),
        (r"\bonly\s+if\b", "condition"),
        (r"\bif\b", "condition_short"),
        (r"\bwhere\b", "condition_short"),
        (r"\bwhen\b", "condition_short"),
    ]

    matches = []
    for pattern, kind in all_signals:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            word = m.group(0).lower()
            start = m.start()
            # For short triggers (if, where, when), filter out non-qualifier usage
            if kind == "condition_short":
                after = text[m.end():m.end() + 15].strip().lower()
                if word == "if" and after.startswith("any"):
                    continue
                before = text[max(0, start - 15):start].strip().lower()
                if before.endswith("as and"):
                    continue

            matches.append((start, m.group(0), "condition" if "condition" in kind else "exception"))

    conditions: List[str] = []
    exceptions: List[str] = []

    if matches:
        matches.sort(key=lambda x: x[0])
        filtered_matches = []
        for m in matches:
            if not filtered_matches:
                filtered_matches.append(m)
            else:
                prev = filtered_matches[-1]
                if m[0] < prev[0] + len(prev[1]):
                    continue
                filtered_matches.append(m)

        for i, (start_idx, trigger_text, kind) in enumerate(filtered_matches):
            if i + 1 < len(filtered_matches):
                next_start = filtered_matches[i + 1][0]
                raw_span = text[start_idx:next_start].strip()
            else:
                raw_span = text[start_idx:].strip()

            for delim in [".", ";", "\n"]:
                d_idx = raw_span.find(delim)
                if d_idx != -1:
                    raw_span = raw_span[:d_idx].strip()

            trigger_lower = trigger_text.lower()
            if trigger_lower in {"where", "when", "if", "only if", "provided that", "provided further that", "notwithstanding"}:
                comma_idx = raw_span.find(",")
                if comma_idx != -1:
                    after_comma = raw_span[comma_idx + 1:].strip()
                    if re.search(r"\b(shall|must|may|will|is|are|a penalty|the exchange|the board|members?)\b", after_comma, re.IGNORECASE):
                        raw_span = raw_span[:comma_idx].strip()

            span = raw_span.rstrip(",;:-").strip()

            if len(span) > 0:
                if kind == "condition":
                    if span not in conditions:
                        conditions.append(span)
                else:
                    if span not in exceptions:
                        exceptions.append(span)

    # Threshold extraction
    threshold = None
    th_match = re.search(
        r"\b(?:(?:Rs\.?|INR|₹)\s*\d+(?:[,\.]\d+)*(?:\s*(?:lakhs?|crores?|thousand|million|billion))?|\d+(?:\.\d+)?\s*(?:%|percent|per\s+cent)|\d+\s*(?:lakhs?|crores?)|total\s+annual\s+revenue|net\s+worth)\b",
        text,
        re.IGNORECASE,
    )
    if th_match:
        threshold = th_match.group(0).strip()

    # Time / deadline extraction
    time_limit = None
    time_match = re.search(
        r"\b(?:within\s+\d+\s*(?:working\s+days?|calendar\s+days?|days?|months?|years?|hours?)|15\s+working\s+days|immediately|forthwith|within\s+\d+\s*days|by\s+\d+(?:st|nd|rd|th)?\s+of\s+[A-Za-z]+|monthly|quarterly|half[\s-]yearly|annually)\b",
        text,
        re.IGNORECASE,
    )
    if time_match:
        time_limit = time_match.group(0).strip()

    # Source circular reference extraction (body or footnote)
    source_circular = None
    circ_match = re.search(
        r"\b(?:SEBI/HO/[A-Za-z0-9_\-]+/[A-Za-z0-9_\-]+/(?:P/)?CIR/\d+/\d+|CIR/[A-Za-z0-9_\-]+/\d+/\d+|HO/\d+/\d+/\d+\([0-9]+\)\d+-[A-Za-z0-9_\-]+/[A-Z]/\d+/\d+|Circular\s+no\.?\s*[A-Za-z0-9\/\-_]+(?:\s+dated\s+[A-Za-z]+\s+\d+,\s+\d+)?)\b",
        text,
        re.IGNORECASE,
    )
    if circ_match:
        source_circular = re.sub(r"^Circular\s+no\.?\s*", "", circ_match.group(0).strip(), flags=re.IGNORECASE)

    return {
        "conditions": conditions,
        "exceptions": exceptions,
        "threshold": threshold,
        "time": time_limit,
        "source_circular": source_circular,
    }
