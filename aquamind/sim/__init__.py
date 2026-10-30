"""The simulated town (TRUTH side): hidden leaks, hydraulics, citizens, records, field crews, the duty operator.

Methods never import this package. They read cases through `aquamind.case.load_public_case` and act through the
environment (`aquamind.sim.field.CaseEnv`), which answers from the hidden truth.
"""
