"""AquaMind demo UI: a control-room web app that shows the system working, for the project presentation.

Everything UI-related lives in this package. It only READS the AquaMind code, benchmark and logs; it never changes them.

    python demo_ui                      # serve on http://127.0.0.1:8765 and open the browser
    python demo_ui record --cases ...   # record full-fidelity investigations for the Mission Control replay

See demo_ui/README.md for the presenter guide.
"""
