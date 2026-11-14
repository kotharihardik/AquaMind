"""The four roles of the AquaMind crew, one module each, plus their prompts and the shared plumbing.

    prompts.py       the role prompts (DOMAIN, FORENSIC, SKEPTIC, STRATEGIST, AUDITOR), kept byte-identical to the logged runs
    base.py          CrewBase: one LLM turn (_ask), the board text an agent reads (_text), the hypothesis slate
    forensic.py      ForensicMixin: Forensic Advocate turn, open questions, witness places, hypotheses as search space
    skeptic.py       SkepticMixin: Critical Skeptic turn
    field_checks.py  FieldCheckMixin: the menu of field checks, their computed chance, the code guard (ADR-009), M3 confirmation
    strategist.py    StrategistMixin: Hydraulic Strategist turn
    auditor.py       AuditorMixin: Safety & SOP Auditor turn

The mixins are combined into `Crew` in aquamind/agent/crew.py, which also holds the switches (__init__) and the investigation loop (run).
"""
from .auditor import AuditorMixin
from .base import CrewBase
from .field_checks import FieldCheckMixin
from .forensic import ForensicMixin
from .prompts import AUDITOR, DOMAIN, FORENSIC, SKEPTIC, STRATEGIST
from .skeptic import SkepticMixin
from .strategist import StrategistMixin

__all__ = ["CrewBase", "ForensicMixin", "SkepticMixin", "FieldCheckMixin", "StrategistMixin", "AuditorMixin",
           "DOMAIN", "FORENSIC", "SKEPTIC", "STRATEGIST", "AUDITOR"]
