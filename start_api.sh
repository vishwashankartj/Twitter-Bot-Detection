#!/bin/bash

# Start the FastAPI server
echo "🚀 Starting Twitter Bot Detection API..."
echo "📚 Swagger docs will be available at: http://localhost:8000/"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the API
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
