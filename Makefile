# Define virtual environment and flask app
VENV = venv
FLASK_APP = app.py

# Define python command (use python3 explicitly)
PYTHON = python3

# Install dependencies
install:
	$(PYTHON) -m venv $(VENV)
	./$(VENV)/bin/pip install --upgrade pip
	./$(VENV)/bin/pip install -r requirements.txt

# Run the Flask application
run:
	FLASK_APP=$(FLASK_APP) FLASK_ENV=development ./$(VENV)/bin/flask run --port 3000

# Clean up virtual environment
clean:
	rm -rf $(VENV)

# Clean up uploaded files
clean-uploads:
	rm -rf uploads/*

# Reinstall all dependencies
reinstall: clean install

# Default target
.DEFAULT_GOAL := install

# Phony targets
.PHONY: install run clean clean-uploads reinstall