.PHONY: all quick smoke test figures provenance clean

all:                     ## full protocol (Appendix B)
	python run_all.py --preset paper --out results
	python scripts/make_provenance.py --out results

quick:
	python run_all.py --preset quick --out results-quick
	python scripts/make_provenance.py --out results-quick

smoke:
	python run_all.py --preset smoke --out results-smoke

test:
	python -m pytest tests -q

figures:
	python run_all.py --preset paper --out results --resume

provenance:
	python scripts/make_provenance.py --out results

clean:
	rm -rf results results-quick results-smoke .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
