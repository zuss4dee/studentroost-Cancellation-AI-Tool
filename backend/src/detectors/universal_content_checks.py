"""
Universal document content checks (mirrors frontend/lib/universalChecks.ts).
Generic heuristics only — no single-document hardcoding.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

CONSUMER_CREATOR_RE = re.compile(
    r"wps|microsoft\s*word|ms\s*word|libreoffice|google\s*docs|docs\.google|pages|openoffice|neooffice|iwork",
    re.I,
)

INSTITUTIONAL_CUES = re.compile(
    r"\.(edu|ac\.[a-z]{2}|gov\.[a-z]{2}|gouv\.|go\.jp|go\.kr|gc\.ca)\b|"
    r"\b(university|college|polytechnic|hospital|clinic|ministry|department|authority|council|agency|"
    r"national\s+health|public\s+health)\b|"
    r"[\u4e00-\u9fff\u3400-\u4dbf](?:\s*[\u4e00-\u9fff\u3400-\u4dbf]){0,20}(?:大学|学院|政府|部|局|医院)",
    re.I,
)

ENGLISH_SPEAKING_COUNTRY_CUES = re.compile(
    r"\b(united\s+kingdom|uk\b|great\s+britain|britain|england|scotland|wales|northern\s+ireland|ireland|"
    r"united\s+states|u\.s\.a\.?|\busa\b|canada|australia|new\s+zealand)\b|"
    r"\.(gov\.uk|ac\.uk|edu\.au|gc\.ca)\b",
    re.I,
)

CHINA_REGION_CUES = re.compile(
    r"\b(china|people'?s\s+republic|prc|beijing|shanghai|guangzhou|shenzhen|hong\s+kong\s+sar|macau|taiwan|r\.o\.c\.)\b|"
    r"\.cn\b|中国|中華|香港|澳門|台灣",
    re.I,
)

PLACEHOLDER_AFTER_LABEL_RE = re.compile(
    r"(Signature|Name\s*\(print\)?|Author|Prepared\s*by|Approved\s*by|Reviewer|Signatory)\s*[:：]\s*"
    r"(N/A|TBC|TBD|\[\s*Name\s*\]|XXXXX?|x{3,}|\.{3,}|—{2,}|-{3,}|\bData\b|\bAdmin\b|\bUser\b|\bOwner\b|\bLorem\s+ipsum\b)",
    re.I,
)

BLANK_AFTER_LABEL_RE = re.compile(
    r"(Signature|Name\s*\(print\)?|Author|Date\s*signed|Signatory)\s*[:：]\s*($|\n|\r|</?[^>]+>)",
    re.I | re.M,
)

LABELED_VALUE_RE = re.compile(
    r"(Reference|Ref\.?|Case\s*(?:ID|No\.?)?|Account\s*(?:No\.?)?|File\s*(?:No\.?)?|ID|Tracking|Confirmation)\s*[:#]?\s*([^\n\r]+)",
    re.I,
)


def _parse_pdf_metadata_date(raw: str) -> Optional[Tuple[datetime, Optional[int]]]:
    """Returns (utc-naive datetime for calendar day, offset_minutes or None)."""
    s = (raw or "").strip()
    if not s:
        return None
    m = re.match(
        r"D:(\d{4})(\d{2})(\d{2})(?:(\d{2})(?:(\d{2})(?:(\d{2}))?)?)?\s*(Z|[+\-]\d{2}'?\d{2}'?)?",
        s,
        re.I,
    )
    if not m:
        return None

    y, mon, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    hh = int(m.group(4) or 0)
    mm = int(m.group(5) or 0)
    ss = int(m.group(6) or 0)
    offset_minutes: Optional[int] = None
    tz = m.group(7)
    if tz and re.match(r"^[+\-]", tz):
        tzm = re.match(r"^([+\-])(\d{2})'?(\d{2})'?$", tz.replace("'", ""))
        if tzm:
            sign = -1 if tzm.group(1) == "-" else 1
            offset_minutes = sign * (int(tzm.group(2)) * 60 + int(tzm.group(3)))
    elif tz and tz.upper() == "Z":
        offset_minutes = 0

    dt = datetime(y, mon, d, hh, mm, ss, tzinfo=timezone.utc)
    return dt, offset_minutes


def _calendar_day_utc(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    d = dt.astimezone(timezone.utc).date()
    return d.toordinal()


def _add_parsed_date(by_day: Dict[int, str], y: int, mon: int, day: int, raw_snippet: str) -> None:
    if y < 1000 or mon < 1 or mon > 12 or day < 1 or day > 31:
        return
    try:
        datetime(y, mon, day, tzinfo=timezone.utc)
    except ValueError:
        return
    key = datetime(y, mon, day, tzinfo=timezone.utc).date().toordinal()
    if key not in by_day:
        by_day[key] = raw_snippet.strip()


def _extract_written_dates(text: str) -> Dict[int, str]:
    by_day: Dict[int, str] = {}
    if not text:
        return by_day

    for m in re.finditer(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text):
        _add_parsed_date(by_day, int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(0))

    for m in re.finditer(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s*(\d{4})\b",
        text,
        re.I,
    ):
        mon_name = m.group(1).lower()
        mon = MONTHS.get(mon_name)
        if mon:
            _add_parsed_date(by_day, int(m.group(3)), mon, int(m.group(2)), m.group(0))

    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        _add_parsed_date(by_day, int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(0))

    for m in re.finditer(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b", text):
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 1900 if y >= 70 else 2000
        if a > 12:
            _add_parsed_date(by_day, y, b, a, m.group(0))
        elif b > 12:
            _add_parsed_date(by_day, y, a, b, m.group(0))
        else:
            _add_parsed_date(by_day, y, b, a, m.group(0))
            if a != b:
                _add_parsed_date(by_day, y, a, b, m.group(0))

    return by_day


def _earliest_written_day(written: Dict[int, str]) -> Optional[Tuple[int, str]]:
    if not written:
        return None
    k = min(written.keys())
    return k, written[k]


def _classify_region_bucket(text: str) -> str:
    tl = text.lower()
    if CHINA_REGION_CUES.search(text) or re.search(r"[\u4e00-\u9fff]{3,}", text):
        if re.search(r"\b(japan|tokyo|osaka|日本|東京)\b", text, re.I):
            return "JP"
        return "CN_HK_TW"
    if re.search(r"\b(japan|tokyo|osaka|日本)\b", text, re.I):
        return "JP"
    if ENGLISH_SPEAKING_COUNTRY_CUES.search(text):
        if re.search(r"\.(gov\.uk|ac\.uk)\b", text, re.I) or re.search(
            r"\b(england|scotland|wales|northern\s+ireland|britain|uk\b)\b", tl
        ):
            return "UK_IE"
        if re.search(r"\b(united\s+states|u\.s\.a\.?|\busa\b)\b", tl):
            return "US_CA_DIVERSE"
        return "EN_WEST"
    return "UNKNOWN"


def _offset_allowed_for_bucket(bucket: str, offset_minutes: Optional[int]) -> bool:
    if offset_minutes is None:
        return True
    o = offset_minutes
    if bucket == "UK_IE":
        return -60 <= o <= 120
    if bucket == "CN_HK_TW":
        return 420 <= o <= 540
    if bucket == "JP":
        return 480 <= o <= 600
    if bucket == "EN_WEST":
        return -600 <= o <= 720
    if bucket == "US_CA_DIVERSE":
        return -600 <= o <= 480
    return True


def _detect_primary_language(text: str) -> str:
    sample = text[:12000]
    cjk = len(re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf]", sample))
    latin = len(re.findall(r"[a-zA-Z]", sample))
    lower = sample.lower()
    english_hits = sum(1 for w in (" the ", " and ", " of ", " to ", " in ", " is ", " for ") if w in lower)

    if cjk > 40 and cjk >= latin * 0.25:
        return "zh"
    if latin > 80 and english_hits >= 3:
        return "en"
    if latin > 40 and cjk < 15:
        return "latin_other"
    if cjk > 15:
        return "zh_mixed"
    return "undetermined"


def _normalize_labeled_value(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


NAME_STOP = frozenset(
    "the and for ltd inc plc llc corp company university college hospital department ministry "
    "council authority page ref date from subject".split()
)


def run_universal_content_checks(
    text: str,
    creator: str,
    producer: str,
    creation_raw: str,
) -> Dict[str, Any]:
    """
    Returns {
      "flags": [ { "type", "severity", "label", "detail" }, ... ],
      "high_severity_count": int,
    }
    """
    flags: List[Dict[str, str]] = []
    t = text or ""
    creator = creator or ""
    creation_raw = creation_raw or ""

    written = _extract_written_dates(t)
    earliest = _earliest_written_day(written)
    pdf_parsed = _parse_pdf_metadata_date(creation_raw)

    institution_hints: List[str] = []
    if INSTITUTIONAL_CUES.search(t):
        institution_hints.append("institutional_cues_in_body")
    if ENGLISH_SPEAKING_COUNTRY_CUES.search(t):
        institution_hints.append("english_speaking_region_cues")
    if CHINA_REGION_CUES.search(t):
        institution_hints.append("china_region_cues")
    if re.search(r"[\u4e00-\u9fff]", t):
        institution_hints.append("contains_cjk_characters")

    claims_institution = bool(INSTITUTIONAL_CUES.search(t)) or (
        "institutional_cues_in_body" in institution_hints
    )

    if earliest and pdf_parsed:
        dt_utc, _ = pdf_parsed
        pdf_day = _calendar_day_utc(dt_utc)
        gap = abs(pdf_day - earliest[0])
        if gap > 3:
            raw_snip = earliest[1]
            if pdf_day > earliest[0]:
                direction = (
                    f"PDF creation is {gap} calendar days after the earliest written date in the body ({raw_snip})."
                )
            else:
                direction = (
                    f"PDF creation is {gap} calendar days before the earliest written date in the body ({raw_snip})."
                )
            flags.append(
                {
                    "type": "date_discrepancy",
                    "severity": "high",
                    "label": "Date discrepancy",
                    "detail": f"Document date does not match PDF creation date — {direction}",
                }
            )

    if creator and CONSUMER_CREATOR_RE.search(creator) and claims_institution:
        flags.append(
            {
                "type": "consumer_software",
                "severity": "high",
                "label": "Consumer authoring software",
                "detail": "Document was created with personal software rather than institutional document systems.",
            }
        )

    bucket = _classify_region_bucket(t)
    if bucket != "UNKNOWN" and pdf_parsed and pdf_parsed[1] is not None:
        if not _offset_allowed_for_bucket(bucket, pdf_parsed[1]):
            flags.append(
                {
                    "type": "timezone_issuer",
                    "severity": "high",
                    "label": "Timezone vs issuer",
                    "detail": "PDF was created in a timezone inconsistent with the claimed issuing institution's location.",
                }
            )

    placeholder_matches = 0
    for _ in PLACEHOLDER_AFTER_LABEL_RE.finditer(t):
        flags.append(
            {
                "type": "placeholder_field",
                "severity": "medium",
                "label": "Placeholder or blank field",
                "detail": "Unfilled or placeholder field detected — document may be incomplete or templated.",
            }
        )
        placeholder_matches += 1
        if placeholder_matches >= 12:
            break
    if placeholder_matches < 12:
        for _ in BLANK_AFTER_LABEL_RE.finditer(t):
            flags.append(
                {
                    "type": "placeholder_field",
                    "severity": "medium",
                    "label": "Placeholder or blank field",
                    "detail": "Unfilled or placeholder field detected — document may be incomplete or templated.",
                }
            )
            placeholder_matches += 1
            if placeholder_matches >= 12:
                break

    by_label: Dict[str, Set[str]] = {}
    for m in LABELED_VALUE_RE.finditer(t):
        key = re.sub(r"\s+", " ", m.group(1)).strip().lower()
        val = _normalize_labeled_value(m.group(2))
        if len(val) < 2:
            continue
        by_label.setdefault(key, set()).add(val)

    for label, vals in by_label.items():
        if len(vals) > 1:
            flags.append(
                {
                    "type": "internal_inconsistency",
                    "severity": "high",
                    "label": "Internal inconsistency",
                    "detail": f"Internal inconsistency detected — {label} appears differently across the document.",
                }
            )

    lang = _detect_primary_language(t)
    uk_english_issuer = bool(
        re.search(r"\.(gov\.uk|ac\.uk|nhs\.uk)\b", t, re.I)
        or re.search(r"\b(nhs|ukvi|home\s*office|britain|united\s+kingdom)\b", t, re.I)
    )
    china_issuer = bool(CHINA_REGION_CUES.search(t))

    if uk_english_issuer and lang not in ("en", "undetermined"):
        flags.append(
            {
                "type": "language_issuer",
                "severity": "high",
                "label": "Language vs issuer",
                "detail": "Document language is inconsistent with the claimed issuing institution's country.",
            }
        )
    if china_issuer and not re.search(r"[\u4e00-\u9fff]", t) and lang not in ("zh", "zh_mixed"):
        flags.append(
            {
                "type": "language_issuer",
                "severity": "high",
                "label": "Language vs issuer",
                "detail": "Document language is inconsistent with the claimed issuing institution's country.",
            }
        )

    high_severity_count = sum(1 for f in flags if f.get("severity") == "high")

    return {
        "flags": flags,
        "high_severity_count": high_severity_count,
    }
