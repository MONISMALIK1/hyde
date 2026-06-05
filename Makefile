.PHONY: help test install clean ask bench

help:		## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

test:		## Run the offline test suite (no API key needed)
	cd .. && python -m unittest discover -s hyde/tests -t . -v

install:	## Editable install of the package
	pip install -e .

ask:		## Answer a question: make ask ARGS='"Why does my money buy less?" --show-trace'
	python -m hyde $(ARGS)

bench:		## Compare retrieval@k of the raw query vs HyDE on the eval set
	python -m hyde --bench

clean:		## Remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf *.egg-info build dist .eggs
