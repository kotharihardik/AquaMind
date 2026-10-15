"""Read times like '14:15', '1415 hrs', '2 PM', '10/10 14:15' or '2018-10-10 14:15' into case seconds.

Work orders are written by many people, so times come in many shapes. If no date is given we take the day that
puts the time closest to a reference moment (usually 'now' or the anomaly time).
"""
from __future__ import annotations

import datetime as dt
import re

MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                      "dec"], start=1)}
_DATE = re.compile(r"(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})|(?P<d2>\d{1,2})/(?P<m2>\d{1,2})"
                   r"|(?P<d3>\d{1,2}) (?P<mon>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.I)
_TIME = re.compile(r"(?P<h>\d{1,2}):(?P<mi>\d{2})\s*(?P<ap>am|pm)?|(?P<hm>\d{3,4})\s*hrs"
                   r"|(?P<h2>\d{1,2})\s*(?P<ap2>am|pm)", re.I)
# "14:15-18:55", "1415 hrs-1855 hrs", "from 2 PM to 6 PM", "14:15 till about 18:55"
_T = r"(?:\d{1,2}:\d{2}\s*(?:am|pm)?|\d{3,4}\s*hrs|\d{1,2}\s*(?:am|pm))"
_RANGE = re.compile(rf"({_T})\s*(?:-|–|to|till about|till|until)\s*({_T})", re.I)


def _clock(text: str) -> tuple[int, int] | None:
    m = _TIME.search(text)
    if not m:
        return None
    if m.group("h") is not None:
        h, mi, ap = int(m.group("h")), int(m.group("mi")), (m.group("ap") or "").lower()
    elif m.group("hm") is not None:
        hm = m.group("hm").zfill(4)
        h, mi, ap = int(hm[:2]), int(hm[2:]), ""
    else:
        h, mi, ap = int(m.group("h2")), 0, m.group("ap2").lower()
    if ap == "pm" and h < 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    return (h, mi) if h < 24 and mi < 60 else None


def _date(text: str, year: int) -> dt.date | None:
    m = _DATE.search(text)
    if not m:
        return None
    try:
        if m.group("y"):
            return dt.date(int(m.group("y")), int(m.group("m")), int(m.group("d")))
        if m.group("d2"):
            return dt.date(year, int(m.group("m2")), int(m.group("d2")))
        return dt.date(year, MONTHS[m.group("mon").lower()], int(m.group("d3")))
    except ValueError:
        return None


def parse_when(text: str, start_dt: dt.datetime, ref_s: float) -> float | None:
    """Text -> seconds since the case start. Without a date: the day closest to ref_s."""
    hm = _clock(str(text))
    if hm is None:
        return None
    day = _date(str(text), start_dt.year)
    if day is not None:
        t = dt.datetime.combine(day, dt.time(*hm))
        return (t - start_dt).total_seconds()
    ref = start_dt + dt.timedelta(seconds=ref_s)
    cands = [dt.datetime.combine(ref.date() + dt.timedelta(days=k), dt.time(*hm)) for k in (-1, 0, 1)]
    best = min(cands, key=lambda c: abs((c - ref).total_seconds()))
    return (best - start_dt).total_seconds()


def find_range(text: str) -> tuple[str, str] | None:
    """'... 10/10 1415 hrs-1855 hrs ...' -> ('10/10 1415 hrs', '10/10 1855 hrs') (date copied to both ends)."""
    m = _RANGE.search(text)
    if not m:
        return None
    d = _DATE.search(text)
    pre = (d.group(0) + " ") if d else ""
    return pre + m.group(1), pre + m.group(2)
