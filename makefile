.PHONY: venv install test clean

# Define the virtual environment directory
VENV_DIR = venv

# Define the test directory
TEST_DIR = tests

# Define the Python interpreter to use for the virtual environment
PYTHON = python3

# Define the requirements file
REQUIREMENTS = requirements.txt

# Target to create the virtual environment
venv:
	if [ -d $(VENV) ]; then \
	    exit 1;
	fi
	$(PYTHON) -m venv $(VENV_DIR)
	. $(VENV_DIR)/bin/activate; pip install --upgrade pip

compile: venv
	pip install poetry poetry-plugin-export
	poetry config warnings.export false
	poetry export -f requirements.txt --output requirements.txt

# Target to install dependencies
install: venv
	. $(VENV_DIR)/bin/pip install -r $(REQUIREMENTS)

# Target to run tests
test: install
	. $(VENV_DIR)/bin/activate; $(PYTHON) -m unittest discover -s $(TEST_DIR)

# Target to clean up the environment
clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
