# Crystal-Ball

🔮 **OCSF AI - Crystal Ball** is an intelligent security data analysis API that leverages AI to provide insights into OCSF (Open Cybersecurity Schema Framework) data.

## Features

- **AI-Powered Analysis**: Advanced natural language processing for security data
- **OCSF Integration**: Native support for Open Cybersecurity Schema Framework
- **Vector Search**: Efficient similarity search across security datasets
- **Interactive Interface**: User-friendly web interface for data exploration
- **Docker Ready**: Containerized deployment for easy scalability

## Quick Start with Docker

```bash
docker pull thewintershadow/ocsf-ai:latest
docker run --gpus all -p 8000:8000 \
  -e HUGGINGFACE_TOKEN=YOUR_TOKEN \
  ocsf-ai
```

## Usage

This tool helps security analysts and researchers:
- Analyze security event data using natural language queries
- Discover patterns and correlations in OCSF datasets
- Generate insights from large volumes of security telemetry
- Explore threat intelligence and security frameworks

## Architecture

- **Backend**: Python-based API with FastAPI
- **AI Engine**: Large Language Model integration for intelligent analysis
- **Data Storage**: Vector database for efficient similarity search
- **Frontend**: Modern web interface for user interaction

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

See LICENSE file for details.
