#!/bin/bash

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

echo "Setup complete"
