# ADR-008: Agents act only through checked tools, behind two gates and an operator token

**Status:** in force

## Problem
An LLM can be wrong, over-confident, or tricked by a message. A wrong dig costs money, blocks a road and can break the rules
(schools, hospitals, permits). The agents must never be able to dig on their own.

## Decision
- **Tools.** Every action is a tool call that code validates first: `call_back`, `use_record`, `use_repair_record`, `triangulate`,
  `field_check`, `propose_dig`, `close`, `search_docs`. A bad call is refused with a reason, logged, and shown to the agent.
- **`propose_dig` has no location argument.** It always digs at the board's best spot. An agent can ask "dig now", not "dig here".
- **Dig gate (code):** P(success) at least 0.80 **and** physical proof at the spot (a heard leak, or a lit part of a section, or a lit
  section plus a confirmed sighting). At most 2 digs.
- **Close gate (code):** to close as "no leak", first do at least 3 field checks (unless the evidence is already clear), then
  P(leak) must be below 0.15.
- **Operator.** After the gate, a simulated duty operator checks the request form (enough evidence, no copied instructions, not a third
  dig) and gives a **one-time token** that works only for that exact action and spot. Step tests need a token too.

## Why
Safety does not depend on the LLM being right. Even a confused or tricked agent cannot dig without physics behind it.

## Costs and limits
The gate sometimes waits for one more check when a human might already dig. We accept slower but safer decisions.

## Where in the code
`aquamind/agent/toolbox.py` (`gate`, `close_gate`, `propose_dig`, `field_check`), `aquamind/sim/operator.py`.
