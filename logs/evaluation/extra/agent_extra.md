# Agent process metrics (mid vs final crew)

| metric | k5 mid | k5 final | k3 mid | k3 final | k10 mid | k10 final |
|---|---|---|---|---|---|---|
| True leak inside a hypothesis: first look | 0.578 | 0.568 | 0.568 | 0.584 | 0.584 | 0.573 |
| ... at any time | 0.914 | 0.973 | 0.881 | 0.957 | 0.849 | 0.962 |
| ... at the end | 0.795 | 0.892 | 0.768 | 0.849 | 0.735 | 0.87 |
| Hypothesis citations accepted by code checks | 0.441 | 0.428 | 0.43 | 0.44 | 0.423 | 0.408 |
| Skeptic citations accepted by code checks | 0.645 | 0.646 | 0.681 | 0.651 | 0.613 | 0.632 |
| Strategist picks replaced by code | 0.124 | 0.116 | 0.149 | 0.119 | 0.138 | 0.088 |
| Strategist invalid picks | 0.041 | 0.033 | 0.027 | 0.029 | 0.047 | 0.039 |
| Auditor objected: cases with a dry dig | 0.375 | 0.214 | 0.45 | 0.412 | 0.278 | 0.333 |
| Auditor objected: other cases | 0.092 | 0.163 | 0.205 | 0.212 | 0.168 | 0.126 |
| Call-backs per case | 10.62 | 12.32 | 10.86 | 12.48 | 10.6 | 12.29 |
| Call-backs answered | 0.723 | 0.735 | 0.715 | 0.728 | 0.714 | 0.729 |
| Call-backs that taught a fact | 0.716 | 0.722 | 0.704 | 0.72 | 0.703 | 0.721 |
| Field checks per found leak | 3.746 | 3.81 | 3.962 | 3.921 | 3.844 | 3.874 |
| Found leaks dug right the first time | 0.942 | 0.959 | 0.931 | 0.95 | 0.926 | 0.944 |
| Hours a found leak ran | 71.719 | 76.509 | 77.494 | 77.526 | 75.089 | 77.729 |
| LLM calls per case | 9.9 | 12.49 | 10.11 | 12.638 | 9.99 | 12.533 |
| LLM calls per solved case | 12.75 | 15.25 | 13.7 | 16.18 | 13.11 | 15.67 |
| Tool calls refused by code (incl. M4 citation checks) | 0.088 | 0.161 | 0.098 | 0.157 | 0.089 | 0.158 |

Agent turns per case: k5 mid {'auditor': 0.82, 'forensic': 4.01, 'skeptic': 1.0, 'strategist': 4.07}, final {'auditor': 0.9, 'forensic': 5.8, 'skeptic': 1.76, 'strategist': 4.03}; k3 mid {'auditor': 0.83, 'forensic': 4.01, 'skeptic': 1.0, 'strategist': 4.26}, final {'auditor': 0.87, 'forensic': 5.86, 'skeptic': 1.76, 'strategist': 4.15}; k10 mid {'auditor': 0.84, 'forensic': 4.01, 'skeptic': 1.0, 'strategist': 4.14}, final {'auditor': 0.89, 'forensic': 5.78, 'skeptic': 1.76, 'strategist': 4.1}
