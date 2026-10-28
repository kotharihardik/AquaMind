"""RQ2 Baseline: Plain LLM Without Tools (Proposal Section 10).

Gets the case as text (sensor summary + raw messages) and answers once:
where to dig, or no dig. Tests what tool calling and the active loop add.
"""
from __future__ import annotations

from .react_agent import format_sensor_summary, resolve_place

PLAIN_SYSTEM = """You are a control-room engineer at a water utility. From the case below decide ONE thing: where to
send a crew to dig for a leak, or that no dig is justified. Name the dig place as a street and/or landmark from
the case, or a junction id. Answer JSON: {"decision": "dig" | "no_dig", "place": "...", "reason": "..."}"""


def _cert(evidence: list[str], why: str) -> dict:
    return {"why": why, "evidence": evidence}


class PlainLLM:
    name = "plain_llm"

    def __init__(self, llm_factory):
        self.llm_factory = llm_factory

    def run_case(self, world, env, case) -> dict:
        llm = self.llm_factory()
        text = (f"Case opened by {case.trigger.replace('_', ' ')}"
                + (f" (message {case.trigger_report})" if getattr(case, "trigger_report", None) else "")
                + f". Sensors: {format_sensor_summary(case)}.\n"
                f"Step-test sections: " + "; ".join(s["label"] for s in world.town.sections.items) + "\n"
                "Citizen messages:\n" + "\n".join(f"- {r['report_id']} ({r['received_at']}): {r['text']}"
                                                  for r in case.reports))
        d = llm.json(PLAIN_SYSTEM, text, ["decision"], tag="plain_llm")
        guess = None
        if d.get("decision") == "dig":
            j = resolve_place(world, d.get("place"))
            if j is not None:
                spot = world.town.net.junctions[j]
                guess = spot
                ap = env.request_approval("dig", spot, _cert([str(d.get("reason", ""))[:200], "LLM judgement"], "plain LLM"))
                if ap.get("approved"):
                    env.dig(spot, ap["token"])
        if not env.done:
            env.close("hand_over", "plain LLM done")
        u = llm.usage
        return {"final_guess": guess, "steps": 1, "tool_calls": 1, "llm_calls": u.calls, "llm_cached": u.cached,
                "llm_seconds": u.seconds, "tokens": u.prompt_tokens + u.output_tokens}
