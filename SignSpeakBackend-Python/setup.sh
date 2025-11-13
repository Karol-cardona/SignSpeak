#!/bin/bash

echo "Creating and activating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete. To run the API, use the following command:"
echo "uvicorn app.main:app --reload"