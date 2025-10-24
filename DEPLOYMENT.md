# Deployment Guide

This guide covers deploying the Latency Spike Investigator to various platforms.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Variables](#environment-variables)
- [Render Deployment](#render-deployment)
- [Hugging Face Spaces Deployment](#hugging-face-spaces-deployment)
- [Docker Deployment](#docker-deployment)
- [Local Development](#local-development)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.8+ (3.11 recommended)
- Git repository with the application code
- API keys for external services (optional for demo mode)

## Environment Variables

### Required for Full Functionality

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `GEMINI_API_KEY` | Google Gemini API key for AI analysis | None | No* |

### Optional Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `NEW_RELIC_API_KEY` | New Relic API key | None | No |
| `DATADOG_API_KEY` | Datadog API key | None | No |
| `DATADOG_APP_KEY` | Datadog application key | None | No |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` | No |
| `SPIKE_THRESHOLD_MS` | Latency threshold in milliseconds | `1000` | No |
| `CORRELATION_WINDOW_MINUTES` | Time window for correlation analysis | `15` | No |
| `CACHE_TTL_SECONDS` | Cache time-to-live in seconds | `300` | No |
| `GEMINI_REQUESTS_PER_MINUTE` | Rate limit for Gemini API | `60` | No |
| `DEBUG` | Enable debug mode | `false` | No |
| `ENVIRONMENT` | Environment name | `development` | No |

*The application runs in demo mode without API keys, using mock data sources.

## Render Deployment

### Automatic Deployment

1. **Fork/Clone Repository**
   ```bash
   git clone <your-repo-url>
   cd latency-spike-investigator
   ```

2. **Connect to Render**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

3. **Configure Service**
   - **Name**: `latency-spike-investigator`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false`

4. **Set Environment Variables**
   - Add your API keys in the Render dashboard
   - Set `ENVIRONMENT=production`

### Manual Configuration

If not using the provided `render.yaml`:

1. Create a new Web Service on Render
2. Use the following settings:
   - **Runtime**: Python 3.11
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: See above
   - **Health Check Path**: `/healthz` (if implemented)

## Hugging Face Spaces Deployment

### Step-by-Step Deployment

1. **Create New Space**
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Streamlit" as SDK

2. **Configure Space**
   - **Space name**: `latency-spike-investigator`
   - **License**: MIT
   - **SDK**: Streamlit
   - **SDK Version**: 1.28.0
   - **Hardware**: CPU basic (free tier)

3. **Upload Files**
   - Upload all application files
   - Rename `README_HF.md` to `README.md`
   - Ensure `app.py` is in the root directory

4. **Configure Secrets**
   - Go to Space Settings → Repository secrets
   - Add your API keys:
     ```
     GEMINI_API_KEY=your_gemini_api_key_here
     NEW_RELIC_API_KEY=your_newrelic_key_here
     DATADOG_API_KEY=your_datadog_key_here
     DATADOG_APP_KEY=your_datadog_app_key_here
     ```

5. **Deploy**
   - The space will automatically build and deploy
   - Check the logs for any errors

### Hugging Face Limitations

- **Memory**: Limited to 16GB RAM
- **CPU**: Shared CPU resources
- **Storage**: Ephemeral storage only
- **Network**: Limited outbound connections
- **Timeout**: 60-second request timeout

## Docker Deployment

### Build and Run Locally

```bash
# Build the image
docker build -t latency-spike-investigator .

# Run with environment variables
docker run -p 8501:8501 \
  -e GEMINI_API_KEY=your_key_here \
  -e ENVIRONMENT=production \
  latency-spike-investigator
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - NEW_RELIC_API_KEY=${NEW_RELIC_API_KEY}
      - DATADOG_API_KEY=${DATADOG_API_KEY}
      - DATADOG_APP_KEY=${DATADOG_APP_KEY}
      - ENVIRONMENT=production
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    volumes:
      - ./data:/app/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

Run with:
```bash
docker-compose up -d
```

## Local Development

### Setup

1. **Clone Repository**
   ```bash
   git clone <repo-url>
   cd latency-spike-investigator
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run Startup Validation**
   ```bash
   python startup_checks.py
   ```

6. **Start Application**
   ```bash
   streamlit run app.py
   ```

### Development Commands

```bash
# Run tests
pytest

# Format code
black .

# Lint code
flake8 .

# Run with debug mode
DEBUG=true streamlit run app.py
```

## Troubleshooting

### Common Issues

#### 1. Application Won't Start

**Symptoms**: Application crashes on startup or shows initialization errors.

**Solutions**:
- Run startup validation: `python startup_checks.py`
- Check Python version: `python --version` (requires 3.8+)
- Verify dependencies: `pip list`
- Check environment variables: `env | grep -E "(GEMINI|DATADOG|NEW_RELIC)"`

#### 2. API Connection Failures

**Symptoms**: "API key invalid" or connection timeout errors.

**Solutions**:
- Verify API keys are correctly set
- Check API key permissions and quotas
- Test network connectivity
- Review rate limiting settings

#### 3. Database Errors

**Symptoms**: SQLite connection errors or permission issues.

**Solutions**:
- Ensure `data/` directory exists and is writable
- Check disk space availability
- Verify SQLite installation: `python -c "import sqlite3; print('OK')"`

#### 4. Memory Issues (Hugging Face Spaces)

**Symptoms**: Application killed due to memory limits.

**Solutions**:
- Reduce cache TTL settings
- Limit correlation window size
- Use smaller datasets for demo
- Consider upgrading to paid tier

#### 5. Port Binding Issues (Render/Docker)

**Symptoms**: "Port already in use" or connection refused errors.

**Solutions**:
- Ensure `PORT` environment variable is set correctly
- Use `0.0.0.0` as host address for containers
- Check firewall settings
- Verify Streamlit configuration

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Environment variable
export DEBUG=true

# Or in .env file
DEBUG=true

# Or in deployment platform
DEBUG=true
```

### Health Checks

The application includes health check endpoints:

- **Streamlit built-in**: `/_stcore/health`
- **Custom endpoint**: `/healthz` (if Flask is available)

Test health check:
```bash
curl http://localhost:8501/_stcore/health
```

### Log Analysis

Check application logs for errors:

```bash
# Local development
streamlit run app.py --logger.level=debug

# Docker
docker logs <container-id>

# Render
Check logs in Render dashboard

# Hugging Face Spaces
Check logs in Space settings
```

### Performance Optimization

For better performance:

1. **Reduce Cache TTL**: Lower `CACHE_TTL_SECONDS` for faster updates
2. **Optimize Correlation Window**: Smaller `CORRELATION_WINDOW_MINUTES` for faster analysis
3. **Rate Limiting**: Adjust `GEMINI_REQUESTS_PER_MINUTE` based on quota
4. **Database Optimization**: Use in-memory SQLite for temporary deployments

### Getting Help

If you encounter issues not covered here:

1. Check the application logs for detailed error messages
2. Run `python startup_checks.py` for system validation
3. Verify all environment variables are correctly set
4. Test with minimal configuration (demo mode)
5. Check platform-specific documentation:
   - [Render Docs](https://render.com/docs)
   - [Hugging Face Spaces Docs](https://huggingface.co/docs/hub/spaces)

### Support

For additional support:
- Review the application logs
- Check the GitHub repository for known issues
- Ensure all dependencies are up to date
- Test in a clean environment