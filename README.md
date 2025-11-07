# Crystal-Ball

🔮 **OCSF AI - Crystal Ball** is an intelligent security data analysis platform that leverages AI to provide insights into OCSF (Open Cybersecurity Schema Framework) data through an interactive Streamlit interface.

## Features

- **Multi-Turn Conversations**: Chat naturally with the AI about OCSF concepts
- **AI-Powered Analysis**: Advanced natural language processing for security data
- **OCSF Integration**: Native support for Open Cybersecurity Schema Framework
- **Vector Search**: Efficient similarity search across security datasets
- **Interactive Web UI**: Modern Streamlit interface with conversation history
- **Source Citations**: See exactly which OCSF documents inform each response
- **Docker Ready**: Containerized deployment for easy scalability

## Quick Start

### Local Development
```bash
# Set your Hugging Face token
export HUGGINGFACE_TOKEN=your_token_here

# Install dependencies
pip install -r requirements.txt

# Run the ingestion (first time only)
python ingest.py

# Start the Streamlit app
./run_streamlit.sh
```

### Docker
```bash
docker pull thewintershadow/ocsf-ai:latest
docker run --gpus all -p 8501:8501 \
  -e HUGGINGFACE_TOKEN=YOUR_TOKEN \
  thewintershadow/ocsf-ai:latest
```

Then open http://localhost:8501 in your browser.

## Usage

This tool helps security analysts and researchers:
- **Ask Questions**: Natural language queries about OCSF schema, events, and best practices
- **Multi-Turn Discussions**: Build on previous questions for deeper understanding
- **Explore Documentation**: AI-powered search through official OCSF documentation
- **Get Sourced Answers**: Every response includes citations to source documents

### Example Questions:
- "What are the core OCSF event classes?"
- "How do I model a network connection event?"
- "What's the difference between OCSF 1.0 and 1.1?"
- "Show me examples of authentication events"

## Architecture

- **Frontend**: Streamlit web interface with chat functionality
- **AI Engine**: Google Gemma 2B model with conversation context
- **Data Storage**: ChromaDB vector database for semantic search
- **Knowledge Base**: Automatically ingested OCSF documentation and examples

## API Mode (Legacy)

The FastAPI backend is still available in `app.py` for programmatic access:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

See LICENSE file for details.
