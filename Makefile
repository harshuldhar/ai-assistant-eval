# ─── AI Assistant Eval — Convenience Commands ────────────────────────────────
# Usage (Windows PowerShell, no make installed):
#   py -m pip install -r requirements.txt   (install)
#   streamlit run ui/app.py                  (run UI)
#   py -m evaluation.runner                  (run eval)
#   py report/generate_report.py             (generate PDF)
#   py -m pytest tests/ -v                   (run tests)
#   py tests/sanity_check.py                 (quick API check)
#
# Usage (if make is installed):
#   make install | make run | make eval | make report | make test | make sanity

.PHONY: run eval report test install setup sanity

install:
	py -m pip install -r requirements.txt

setup:
	py -m pip install -r requirements.txt

run:
	streamlit run ui/app.py

eval:
	py -m evaluation.runner

report:
	py report/generate_report.py

test:
	py -m pytest tests/ -v

sanity:
	py tests/sanity_check.py
