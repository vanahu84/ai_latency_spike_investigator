"""
Startup validation and health checks for the Latency Spike Investigator.
"""

import os
import sys
import sqlite3
import requests
from typing import Dict, List, Tuple
from pathlib import Path
import logging

from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StartupValidator:
    """Validates system configuration and dependencies on startup."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def validate_environment(self) -> bool:
        """Validate environment variables and configuration."""
        logger.info("Validating environment configuration...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.errors.append(f"Python 3.8+ required, found {sys.version}")
            
        # Check required environment variables
        missing_config = config.validate_required_config()
        if missing_config:
            self.warnings.append(f"Missing optional API keys: {', '.join(missing_config)}")
            logger.warning("Some API keys are missing - application will run in demo mode")
        
        # Validate numeric configurations
        try:
            threshold = float(config.SPIKE_THRESHOLD_MS)
            if threshold <= 0:
                self.errors.append("SPIKE_THRESHOLD_MS must be positive")
        except ValueError:
            self.errors.append("SPIKE_THRESHOLD_MS must be a valid number")
            
        try:
            window = int(config.CORRELATION_WINDOW_MINUTES)
            if window <= 0:
                self.errors.append("CORRELATION_WINDOW_MINUTES must be positive")
        except ValueError:
            self.errors.append("CORRELATION_WINDOW_MINUTES must be a valid integer")
            
        # Check port configuration
        try:
            port = int(config.PORT)
            if not (1024 <= port <= 65535):
                self.warnings.append(f"Port {port} may not be accessible")
        except ValueError:
            self.errors.append("PORT must be a valid integer")
            
        return len(self.errors) == 0
    
    def validate_database(self) -> bool:
        """Validate database connectivity and setup."""
        logger.info("Validating database configuration...")
        
        try:
            # Ensure data directory exists
            db_path = Path(config.SQLITE_DB_PATH)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Test SQLite connection
            conn = sqlite3.connect(config.SQLITE_DB_PATH)
            conn.execute("SELECT 1")
            conn.close()
            
            logger.info(f"Database accessible at {config.SQLITE_DB_PATH}")
            return True
            
        except Exception as e:
            self.errors.append(f"Database validation failed: {str(e)}")
            return False
    
    def validate_external_apis(self) -> bool:
        """Validate external API connectivity (non-blocking)."""
        logger.info("Validating external API connectivity...")
        
        api_status = {}
        
        # Test Gemini API if configured
        if config.GEMINI_API_KEY:
            try:
                # Simple API validation (without actual request to avoid quota usage)
                if len(config.GEMINI_API_KEY) < 10:
                    self.warnings.append("Gemini API key appears invalid")
                    api_status['gemini'] = False
                else:
                    api_status['gemini'] = True
                    logger.info("Gemini API key configured")
            except Exception as e:
                self.warnings.append(f"Gemini API validation failed: {str(e)}")
                api_status['gemini'] = False
        else:
            self.warnings.append("Gemini API key not configured - AI features disabled")
            api_status['gemini'] = False
        
        # Test Redis connectivity if not using default localhost
        if config.REDIS_URL != "redis://localhost:6379":
            try:
                # Basic URL validation
                if not config.REDIS_URL.startswith(('redis://', 'rediss://')):
                    self.warnings.append("Redis URL format may be invalid")
                    api_status['redis'] = False
                else:
                    api_status['redis'] = True
                    logger.info("Redis URL configured")
            except Exception as e:
                self.warnings.append(f"Redis validation failed: {str(e)}")
                api_status['redis'] = False
        else:
            self.warnings.append("Using default Redis configuration")
            api_status['redis'] = True
        
        # Log API status
        working_apis = sum(1 for status in api_status.values() if status)
        total_apis = len(api_status)
        logger.info(f"API connectivity: {working_apis}/{total_apis} services available")
        
        return True  # Non-blocking validation
    
    def validate_dependencies(self) -> bool:
        """Validate required Python packages."""
        logger.info("Validating Python dependencies...")
        
        required_packages = [
            'streamlit',
            'requests', 
            'redis',
            'google.generativeai',
            'pandas',
            'plotly',
            'dotenv'  # python-dotenv imports as 'dotenv'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.errors.append(f"Missing required packages: {', '.join(missing_packages)}")
            return False
            
        logger.info("All required dependencies available")
        return True
    
    def run_all_checks(self) -> Tuple[bool, Dict[str, List[str]]]:
        """Run all startup validation checks."""
        logger.info("Starting system validation...")
        
        checks = [
            self.validate_environment,
            self.validate_dependencies,
            self.validate_database,
            self.validate_external_apis
        ]
        
        success = True
        for check in checks:
            try:
                if not check():
                    success = False
            except Exception as e:
                self.errors.append(f"Validation check failed: {str(e)}")
                success = False
        
        # Log results
        if self.errors:
            logger.error(f"Validation errors: {'; '.join(self.errors)}")
        if self.warnings:
            logger.warning(f"Validation warnings: {'; '.join(self.warnings)}")
            
        if success:
            logger.info("✅ System validation completed successfully")
        else:
            logger.error("❌ System validation failed")
        
        return success, {
            'errors': self.errors,
            'warnings': self.warnings
        }


def create_health_check_endpoint():
    """Create a simple health check endpoint for deployment platforms."""
    try:
        from flask import Flask, jsonify
        from datetime import datetime
        
        app = Flask(__name__)
        
        @app.route('/healthz')
        def health_check():
            """Health check endpoint for deployment platforms."""
            validator = StartupValidator()
            success, results = validator.run_all_checks()
            
            status_code = 200 if success else 503
            return jsonify({
                'status': 'healthy' if success else 'unhealthy',
                'timestamp': str(datetime.now()),
                'errors': results['errors'],
                'warnings': results['warnings']
            }), status_code
        
        return app
    except ImportError:
        logger.warning("Flask not available - health check endpoint disabled")
        return None


if __name__ == "__main__":
    # Run validation checks
    validator = StartupValidator()
    success, results = validator.run_all_checks()
    
    if not success:
        print("❌ Startup validation failed!")
        for error in results['errors']:
            print(f"  ERROR: {error}")
        sys.exit(1)
    else:
        print("✅ Startup validation passed!")
        if results['warnings']:
            print("Warnings:")
            for warning in results['warnings']:
                print(f"  WARNING: {warning}")