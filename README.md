# Latency Spike Root Cause Investigator

An MVP web application that automatically detects API latency spikes and provides rapid diagnosis through correlation of multiple monitoring data sources.

## Project Structure

```
├── app.py                 # Main Streamlit application entry point
├── config.py             # Configuration management
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── models/              # Data models
├── services/            # Core business logic services
├── clients/             # MCP client interfaces
├── ui/                  # Streamlit UI components
├── tests/               # Test suite
└── data/                # SQLite database storage
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   ```bash
   # On Windows (PowerShell):
   venv\Scripts\Activate.ps1
   
   # On Windows (CMD):
   venv\Scripts\activate.bat
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy environment variables:
   ```bash
   cp .env.example .env
   ```

5. Configure your API keys in the `.env` file

6. Run the application:
   ```bash
   streamlit run app.py
   ```

## Deployment

### Render
- Set environment variables in Render dashboard
- Deploy from this repository

### Hugging Face Spaces
- Upload files to Spaces repository
- Configure secrets for API keys

## Requirements

- Python 3.11+
- Redis (for caching)
- API keys for monitoring services (New Relic, Datadog, Gemini)