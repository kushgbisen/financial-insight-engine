#!/bin/bash

# Setup Script for Financial News RAG component
echo "=== Setting up Financial News RAG for Financial Insight Engine ==="

# Get the project root directory
PROJECT_ROOT=$(pwd)
echo "Project Root: $PROJECT_ROOT"

# Check if we're in the project root directory
if [ ! -d "pages" ]; then
    echo "❌ Error: This script should be run from the project root directory."
    echo "Please navigate to the project root directory and try again."
    exit 1
fi

# Install required packages
echo "Installing RAG-specific dependencies..."
pip install -q faiss-cpu google-generativeai pyarrow python-dotenv

# Create embeddings directory if it doesn't exist
mkdir -p embeddings

# Check for embeddings.zip
if [ -f "embeddings.zip" ]; then
    echo "Found embeddings.zip, extracting to embeddings directory..."
    unzip -o embeddings.zip -d embeddings/
    echo "Extraction complete!"
else
    echo "❗ No embeddings.zip found."
    echo "Please download the embeddings archive and place it in the project root."
    echo "Then run: unzip embeddings.zip -d embeddings/"
fi

# Create or update .env file for API key
if [ ! -f ".env" ]; then
    echo "Creating .env file for API keys..."
    echo "# API keys for Financial Insight Engine" > .env
    echo "GEMINI_API_KEY=AIzaSyA-mK1jrqAviupL63cevB1uvGiIHYB1xcI" >> .env
    echo ".env file created. Please add your Gemini API key."
else
    if ! grep -q "GEMINI_API_KEY" .env; then
        echo "Adding Gemini API key entry to .env file..."
        echo "GEMINI_API_KEY=AIzaSyA-mK1jrqAviupL63cevB1uvGiIHYB1xcI" >> .env
    fi
fi


echo ""
echo "=== RAG Component Setup Complete! ==="
echo "To use the RAG functionality:"
echo "1. Make sure your embeddings are in the 'embeddings' folder"
echo "2. Add your Gemini API key to the .env file"
echo "3. Launch the app with: streamlit run pages/0_🏠_Home.py"
echo "4. Access the RAG component from the sidebar menu"
echo ""