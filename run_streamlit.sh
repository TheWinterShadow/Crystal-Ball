#!/bin/bash

echo "🔮 Starting OCSF AI - Crystal Ball Streamlit App..."
echo "Make sure you have set your HUGGINGFACE_TOKEN environment variable!"
echo ""

# Check if the vector database exists
if [ ! -d "vectorstore" ]; then
    echo "⚠️  Vector database not found. Running ingestion first..."
    python ingest.py
    echo "✅ Vector database created."
    echo ""
fi

# Run Streamlit app
echo "🚀 Starting Streamlit app on http://localhost:8501"
streamlit run streamlit_app.py --server.port=8501 --server.address=localhost