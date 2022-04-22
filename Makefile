VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

algo = "policy_iteration"
gamma = 0.95
threshold = 0.001

run-taxi: $(VENV)/bin/activate
	$(PYTHON) src/test_taxi.py --algo=$(algo) --gamma=$(gamma) --threshold=$(threshold)

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

clean:
	rm -rf src/__pycache__
	rm -rf $(VENV)