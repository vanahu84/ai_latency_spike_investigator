# Troubleshooting Guide

This guide helps resolve common issues when deploying and running the Latency Spike Investigator.

## Quick Diagnostics

### 1. Run System Validation

```bash
python startup_checks.py
```

This will check:
- Python version compatibility
- Required dependencies
- Environment configuration
- Database connectivity
- API key validation

### 2. Check Application Logs

**Local Development:**
```bash
streamlit run app.py --logger.level=debug
```

**Docker:**
```bash
docker logs <container-name>
```

**Render:**
Check the logs in your Render dashboard under "Logs" tab.

**Hugging Face Spaces:**
Check the logs in your Space settings under "Logs" section.

## Common Issues and Solutions

### Startup Issues

#### Issue: "ModuleNotFoundError" on startup

**Cause:** Missing Python dependencies

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or in virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Issue: "Permission denied" database errors

**Cause:** Insufficient permissions for SQLite database

**Solution:**
```bash
# Create data directory with proper permissions
mkdir -p data
chmod 755 data

# Or use in-memory database for testing
export SQLITE_DB_PATH=":memory:"
```

#### Issue: Application crashes with "Port already in use"

**Cause:** Port conflict with another service

**Solution:**
```bash
# Use different port
export PORT=8502
streamlit run app.py --server.port=8502

# Or kill existing process
lsof -ti:8501 | xargs kill -9  # Linux/Mac
netstat -ano | findstr :8501   # Windows - find PID and kill
```

### API Integration Issues

#### Issue: "Invalid API key" errors

**Cause:** Incorrect or missing API keys

**Solution:**
1. Verify API key format and permissions
2. Check environment variable names:
   ```bash
   echo $GEMINI_API_KEY
   echo $NEW_RELIC_API_KEY
   echo $DATADOG_API_KEY
   ```
3. Test API connectivity:
   ```bash
   curl -H "Authorization: Bearer $GEMINI_API_KEY" \
        https://generativelanguage.googleapis.com/v1/models
   ```

#### Issue: "Rate limit exceeded" errors

**Cause:** Too many API requests

**Solution:**
1. Reduce request frequency:
   ```bash
   export GEMINI_REQUESTS_PER_MINUTE=30
   ```
2. Increase cache TTL:
   ```bash
   export CACHE_TTL_SECONDS=600
   ```
3. Use demo mode temporarily:
   ```bash
   unset GEMINI_API_KEY  # Forces demo mode
   ```

### Memory and Performance Issues

#### Issue: Application killed due to memory limits (Hugging Face Spaces)

**Cause:** Insufficient memory allocation

**Solution:**
1. Reduce correlation window:
   ```bash
   export CORRELATION_WINDOW_MINUTES=5
   ```
2. Lower cache settings:
   ```bash
   export CACHE_TTL_SECONDS=60
   ```
3. Use smaller datasets in demo mode
4. Consider upgrading to paid Hugging Face tier

#### Issue: Slow response times

**Cause:** Network latency or resource constraints

**Solution:**
1. Enable caching:
   ```bash
   export REDIS_URL=redis://localhost:6379
   ```
2. Optimize thresholds:
   ```bash
   export SPIKE_THRESHOLD_MS=500  # Lower threshold
   ```
3. Use async processing where possible

### Deployment Platform Issues

#### Render Deployment Issues

**Issue: Build fails with dependency errors**

**Solution:**
1. Check Python version in render.yaml:
   ```yaml
   env: python
   buildCommand: pip install -r requirements.txt
   ```
2. Add system dependencies if needed:
   ```yaml
   buildCommand: apt-get update && apt-get install -y gcc && pip install -r requirements.txt
   ```

**Issue: Health check failures**

**Solution:**
1. Verify health check path in render.yaml:
   ```yaml
   healthCheckPath: /_stcore/health
   ```
2. Or disable health checks temporarily:
   ```yaml
   # healthCheckPath: /healthz
   ```

#### Hugging Face Spaces Issues

**Issue: Space fails to build**

**Cause:** Incompatible dependencies or configuration

**Solution:**
1. Check README.md header format:
   ```yaml
   ---
   title: Latency Spike Investigator
   emoji: 🔍
   colorFrom: blue
   colorTo: red
   sdk: streamlit
   sdk_version: 1.28.0
   app_file: app.py
   ---
   ```
2. Verify file structure:
   ```
   /
   ├── app.py
   ├── requirements.txt
   ├── README.md
   └── [other files]
   ```

**Issue: Secrets not working**

**Solution:**
1. Check secret names match environment variables
2. Restart the Space after adding secrets
3. Verify secrets in Space settings → Repository secrets

#### Docker Issues

**Issue: Container exits immediately**

**Cause:** Application startup failure

**Solution:**
1. Run with interactive mode for debugging:
   ```bash
   docker run -it latency-spike-investigator /bin/bash
   ```
2. Check container logs:
   ```bash
   docker logs <container-id>
   ```
3. Verify environment variables:
   ```bash
   docker run -e DEBUG=true latency-spike-investigator
   ```

**Issue: Cannot connect to external services**

**Cause:** Network configuration or firewall

**Solution:**
1. Test network connectivity:
   ```bash
   docker run --rm latency-spike-investigator curl -I https://google.com
   ```
2. Use host networking for testing:
   ```bash
   docker run --network=host latency-spike-investigator
   ```

### Configuration Issues

#### Issue: Environment variables not loaded

**Cause:** Missing .env file or incorrect format

**Solution:**
1. Create .env file from template:
   ```bash
   cp .env.example .env
   # Edit .env with your values
   ```
2. Verify .env format (no spaces around =):
   ```
   GEMINI_API_KEY=your_key_here
   DEBUG=true
   ```
3. Check file permissions:
   ```bash
   chmod 600 .env
   ```

#### Issue: Database connection errors

**Cause:** SQLite path or permissions issues

**Solution:**
1. Use absolute path:
   ```bash
   export SQLITE_DB_PATH=/app/data/latency_investigator.db
   ```
2. Create directory structure:
   ```bash
   mkdir -p $(dirname $SQLITE_DB_PATH)
   ```
3. Use in-memory database for testing:
   ```bash
   export SQLITE_DB_PATH=":memory:"
   ```

## Debug Mode

Enable debug mode for detailed logging:

```bash
export DEBUG=true
streamlit run app.py
```

Debug mode provides:
- Detailed error messages
- API request/response logging
- Database query logging
- Performance metrics

## Testing Connectivity

### Test API Endpoints

```bash
# Test Gemini API
python -c "
import os
from google.generativeai import configure, GenerativeModel
configure(api_key=os.getenv('GEMINI_API_KEY'))
model = GenerativeModel('gemini-pro')
print('Gemini API: OK')
"

# Test New Relic API
curl -H "Api-Key: $NEW_RELIC_API_KEY" \
     https://api.newrelic.com/v2/applications.json

# Test Datadog API
curl -H "DD-API-KEY: $DATADOG_API_KEY" \
     -H "DD-APPLICATION-KEY: $DATADOG_APP_KEY" \
     https://api.datadoghq.com/api/v1/validate
```

### Test Database

```bash
python -c "
import sqlite3
from config import config
conn = sqlite3.connect(config.SQLITE_DB_PATH)
conn.execute('SELECT 1')
print('Database: OK')
conn.close()
"
```

### Test Redis

```bash
python -c "
import redis
from config import config
r = redis.from_url(config.REDIS_URL)
r.ping()
print('Redis: OK')
"
```

## Performance Optimization

### Memory Usage

1. **Monitor memory usage:**
   ```bash
   # Linux/Mac
   ps aux | grep streamlit
   
   # Docker
   docker stats <container-name>
   ```

2. **Optimize settings:**
   ```bash
   export CACHE_TTL_SECONDS=60          # Shorter cache
   export CORRELATION_WINDOW_MINUTES=5   # Smaller window
   export GEMINI_REQUESTS_PER_MINUTE=30  # Lower rate limit
   ```

### Response Time

1. **Enable caching:**
   ```bash
   export REDIS_URL=redis://localhost:6379
   ```

2. **Optimize thresholds:**
   ```bash
   export SPIKE_THRESHOLD_MS=500  # More sensitive detection
   ```

3. **Use async processing:**
   - Enable auto-refresh with longer intervals
   - Use background processing for heavy operations

## Getting Additional Help

### Log Analysis

1. **Enable verbose logging:**
   ```bash
   export DEBUG=true
   export STREAMLIT_LOGGER_LEVEL=debug
   ```

2. **Check specific components:**
   ```bash
   python -c "
   from startup_checks import StartupValidator
   validator = StartupValidator()
   success, results = validator.run_all_checks()
   print('Errors:', results['errors'])
   print('Warnings:', results['warnings'])
   "
   ```

### System Information

Collect system information for support:

```bash
# System info
python --version
pip list
env | grep -E "(GEMINI|DATADOG|NEW_RELIC|REDIS|SQLITE)"

# Application info
python startup_checks.py
python -c "from config import config; print(f'Config valid: {len(config.validate_required_config()) == 0}')"
```

### Reset to Clean State

If all else fails, reset to a clean state:

```bash
# Remove virtual environment
rm -rf venv

# Remove cache and database
rm -rf data/
rm -rf __pycache__/
rm -rf .pytest_cache/

# Reinstall
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Test basic functionality
python startup_checks.py
streamlit run app.py
```

## Contact and Support

For issues not covered in this guide:

1. **Check the logs** for specific error messages
2. **Run validation** with `python startup_checks.py`
3. **Test in demo mode** by removing API keys
4. **Try minimal configuration** with default settings
5. **Check platform documentation** for deployment-specific issues

Remember to never share API keys or sensitive information when seeking help.