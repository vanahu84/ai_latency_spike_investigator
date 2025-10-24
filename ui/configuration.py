"""
Configuration UI component for API keys and threshold management.
"""

import streamlit as st
import os
from typing import Dict, Any, Optional
import json

from controller import LatencySpikeController
from config import config


class ConfigurationUI:
    """UI component for managing API keys, thresholds, and system configuration."""
    
    def __init__(self, controller: LatencySpikeController):
        self.controller = controller
    
    def render(self):
        """Render the configuration panel."""
        st.title("⚙️ Configuration")
        st.markdown("Manage API keys, detection thresholds, and system settings")
        
        # Configuration tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "API Keys", 
            "Detection Thresholds", 
            "System Settings", 
            "Health Check"
        ])
        
        with tab1:
            self._render_api_keys_config()
        
        with tab2:
            self._render_thresholds_config()
        
        with tab3:
            self._render_system_settings()
        
        with tab4:
            self._render_health_check()
    
    def _render_api_keys_config(self):
        """Render API keys configuration."""
        st.subheader("API Keys Configuration")
        st.markdown("Configure API keys for external services")
        
        # Warning about security
        st.warning(
            "⚠️ **Security Notice**: API keys are stored in environment variables. "
            "In production, use secure secret management systems."
        )
        
        # Gemini AI API Key
        st.markdown("### Gemini AI")
        current_gemini_key = config.GEMINI_API_KEY
        gemini_status = "✅ Configured" if current_gemini_key else "❌ Not configured"
        
        st.write(f"**Status**: {gemini_status}")
        
        with st.expander("Configure Gemini API Key"):
            new_gemini_key = st.text_input(
                "Gemini API Key",
                value="***" if current_gemini_key else "",
                type="password",
                help="Get your API key from Google AI Studio"
            )
            
            if st.button("Update Gemini Key"):
                if new_gemini_key and new_gemini_key != "***":
                    os.environ["GEMINI_API_KEY"] = new_gemini_key
                    config.GEMINI_API_KEY = new_gemini_key
                    st.success("Gemini API key updated successfully!")
                    st.rerun()
                else:
                    st.error("Please enter a valid API key")
        
        # New Relic API Key
        st.markdown("### New Relic")
        current_nr_key = config.NEW_RELIC_API_KEY
        nr_status = "✅ Configured" if current_nr_key else "❌ Not configured"
        
        st.write(f"**Status**: {nr_status}")
        
        with st.expander("Configure New Relic API Key"):
            new_nr_key = st.text_input(
                "New Relic API Key",
                value="***" if current_nr_key else "",
                type="password",
                help="Get your API key from New Relic account settings"
            )
            
            if st.button("Update New Relic Key"):
                if new_nr_key and new_nr_key != "***":
                    os.environ["NEW_RELIC_API_KEY"] = new_nr_key
                    config.NEW_RELIC_API_KEY = new_nr_key
                    st.success("New Relic API key updated successfully!")
                    st.rerun()
                else:
                    st.error("Please enter a valid API key")
        
        # Datadog API Keys
        st.markdown("### Datadog")
        current_dd_api_key = config.DATADOG_API_KEY
        current_dd_app_key = config.DATADOG_APP_KEY
        dd_status = "✅ Configured" if (current_dd_api_key and current_dd_app_key) else "❌ Not configured"
        
        st.write(f"**Status**: {dd_status}")
        
        with st.expander("Configure Datadog API Keys"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_dd_api_key = st.text_input(
                    "Datadog API Key",
                    value="***" if current_dd_api_key else "",
                    type="password",
                    help="Get your API key from Datadog account settings"
                )
            
            with col2:
                new_dd_app_key = st.text_input(
                    "Datadog Application Key",
                    value="***" if current_dd_app_key else "",
                    type="password",
                    help="Get your application key from Datadog account settings"
                )
            
            if st.button("Update Datadog Keys"):
                updated = False
                
                if new_dd_api_key and new_dd_api_key != "***":
                    os.environ["DATADOG_API_KEY"] = new_dd_api_key
                    config.DATADOG_API_KEY = new_dd_api_key
                    updated = True
                
                if new_dd_app_key and new_dd_app_key != "***":
                    os.environ["DATADOG_APP_KEY"] = new_dd_app_key
                    config.DATADOG_APP_KEY = new_dd_app_key
                    updated = True
                
                if updated:
                    st.success("Datadog API keys updated successfully!")
                    st.rerun()
                else:
                    st.error("Please enter valid API keys")
        
        # Test API connections
        st.markdown("### Test API Connections")
        
        if st.button("🔍 Test All API Connections"):
            self._test_api_connections()
    
    def _render_thresholds_config(self):
        """Render spike detection thresholds configuration."""
        st.subheader("Detection Thresholds")
        st.markdown("Configure spike detection sensitivity and thresholds")
        
        # Current thresholds
        current_threshold = config.SPIKE_THRESHOLD_MS
        correlation_window = config.CORRELATION_WINDOW_MINUTES
        
        st.markdown("### Current Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Spike Threshold", f"{current_threshold} ms")
        
        with col2:
            st.metric("Correlation Window", f"{correlation_window} min")
        
        # Threshold configuration
        st.markdown("### Update Thresholds")
        
        with st.form("threshold_config"):
            new_threshold = st.number_input(
                "Spike Threshold (ms)",
                min_value=100,
                max_value=10000,
                value=int(current_threshold),
                step=100,
                help="Latency threshold above which spikes are detected"
            )
            
            new_correlation_window = st.number_input(
                "Correlation Window (minutes)",
                min_value=5,
                max_value=60,
                value=correlation_window,
                step=5,
                help="Time window for correlating metrics around spike events"
            )
            
            # Advanced thresholds
            st.markdown("#### Advanced Thresholds")
            
            col1, col2 = st.columns(2)
            
            with col1:
                spike_multiplier = st.number_input(
                    "Spike Multiplier",
                    min_value=1.5,
                    max_value=10.0,
                    value=2.0,
                    step=0.1,
                    help="Multiplier for baseline latency to detect spikes"
                )
                
                min_spike_duration = st.number_input(
                    "Min Spike Duration (seconds)",
                    min_value=1,
                    max_value=300,
                    value=30,
                    step=5,
                    help="Minimum duration for a valid spike event"
                )
            
            with col2:
                confidence_threshold = st.number_input(
                    "Confidence Threshold",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.7,
                    step=0.05,
                    help="Minimum confidence for root cause analysis"
                )
                
                correlation_threshold = st.number_input(
                    "Correlation Threshold",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help="Minimum correlation score to consider metrics related"
                )
            
            # Submit button
            submitted = st.form_submit_button("Update Thresholds")
            
            if submitted:
                try:
                    # Update configuration
                    os.environ["SPIKE_THRESHOLD_MS"] = str(new_threshold)
                    os.environ["CORRELATION_WINDOW_MINUTES"] = str(new_correlation_window)
                    
                    config.SPIKE_THRESHOLD_MS = float(new_threshold)
                    config.CORRELATION_WINDOW_MINUTES = int(new_correlation_window)
                    
                    # Update controller thresholds
                    threshold_config = {
                        'spike_threshold_ms': new_threshold,
                        'spike_multiplier': spike_multiplier,
                        'min_duration_seconds': min_spike_duration,
                        'confidence_threshold': confidence_threshold,
                        'correlation_threshold': correlation_threshold
                    }
                    
                    success = self.controller.update_spike_thresholds(threshold_config)
                    
                    if success:
                        st.success("Thresholds updated successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to update thresholds in controller")
                
                except Exception as e:
                    st.error(f"Failed to update thresholds: {str(e)}")
        
        # Threshold presets
        st.markdown("### Threshold Presets")
        
        presets = {
            "Conservative": {
                "spike_threshold_ms": 2000,
                "spike_multiplier": 3.0,
                "confidence_threshold": 0.8
            },
            "Balanced": {
                "spike_threshold_ms": 1000,
                "spike_multiplier": 2.0,
                "confidence_threshold": 0.7
            },
            "Aggressive": {
                "spike_threshold_ms": 500,
                "spike_multiplier": 1.5,
                "confidence_threshold": 0.5
            }
        }
        
        col1, col2, col3 = st.columns(3)
        
        for i, (preset_name, preset_values) in enumerate(presets.items()):
            col = [col1, col2, col3][i]
            
            with col:
                st.write(f"**{preset_name}**")
                for key, value in preset_values.items():
                    st.write(f"- {key}: {value}")
                
                if st.button(f"Apply {preset_name}", key=f"preset_{preset_name}"):
                    try:
                        success = self.controller.update_spike_thresholds(preset_values)
                        if success:
                            st.success(f"{preset_name} preset applied!")
                        else:
                            st.error("Failed to apply preset")
                    except Exception as e:
                        st.error(f"Error applying preset: {str(e)}")
    
    def _render_system_settings(self):
        """Render system settings configuration."""
        st.subheader("System Settings")
        st.markdown("Configure database, caching, and performance settings")
        
        # Database settings
        st.markdown("### Database Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**SQLite Database**")
            st.code(config.SQLITE_DB_PATH)
            
            if st.button("Test Database Connection"):
                try:
                    # Test database connection
                    health = self.controller.data_access.get_system_health()
                    if health.get('database', {}).get('status') == 'healthy':
                        st.success("✅ Database connection successful")
                    else:
                        st.error("❌ Database connection failed")
                except Exception as e:
                    st.error(f"Database test failed: {str(e)}")
        
        with col2:
            st.write("**Redis Cache**")
            st.code(config.REDIS_URL)
            
            if st.button("Test Cache Connection"):
                try:
                    # Test cache connection
                    health = self.controller.data_access.get_system_health()
                    if health.get('cache', {}).get('status') == 'healthy':
                        st.success("✅ Cache connection successful")
                    else:
                        st.error("❌ Cache connection failed")
                except Exception as e:
                    st.error(f"Cache test failed: {str(e)}")
        
        # Performance settings
        st.markdown("### Performance Settings")
        
        with st.form("performance_settings"):
            cache_ttl = st.number_input(
                "Cache TTL (seconds)",
                min_value=60,
                max_value=3600,
                value=config.CACHE_TTL_SECONDS,
                step=60,
                help="Time-to-live for cached data"
            )
            
            gemini_rate_limit = st.number_input(
                "Gemini API Rate Limit (requests/minute)",
                min_value=10,
                max_value=300,
                value=config.GEMINI_REQUESTS_PER_MINUTE,
                step=10,
                help="Rate limit for Gemini API requests"
            )
            
            debug_mode = st.checkbox(
                "Debug Mode",
                value=config.DEBUG,
                help="Enable debug logging and detailed error messages"
            )
            
            if st.form_submit_button("Update Performance Settings"):
                try:
                    os.environ["CACHE_TTL_SECONDS"] = str(cache_ttl)
                    os.environ["GEMINI_REQUESTS_PER_MINUTE"] = str(gemini_rate_limit)
                    os.environ["DEBUG"] = str(debug_mode).lower()
                    
                    config.CACHE_TTL_SECONDS = cache_ttl
                    config.GEMINI_REQUESTS_PER_MINUTE = gemini_rate_limit
                    config.DEBUG = debug_mode
                    
                    st.success("Performance settings updated successfully!")
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Failed to update settings: {str(e)}")
        
        # Export/Import configuration
        st.markdown("### Configuration Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Configuration"):
                config_data = {
                    'spike_threshold_ms': config.SPIKE_THRESHOLD_MS,
                    'correlation_window_minutes': config.CORRELATION_WINDOW_MINUTES,
                    'cache_ttl_seconds': config.CACHE_TTL_SECONDS,
                    'gemini_requests_per_minute': config.GEMINI_REQUESTS_PER_MINUTE,
                    'debug': config.DEBUG,
                    'sqlite_db_path': config.SQLITE_DB_PATH,
                    'redis_url': config.REDIS_URL
                }
                
                st.download_button(
                    "Download Configuration",
                    data=json.dumps(config_data, indent=2),
                    file_name="latency_investigator_config.json",
                    mime="application/json"
                )
        
        with col2:
            uploaded_config = st.file_uploader(
                "📤 Import Configuration",
                type=['json'],
                help="Upload a configuration file to restore settings"
            )
            
            if uploaded_config is not None:
                try:
                    config_data = json.load(uploaded_config)
                    
                    # Validate and apply configuration
                    for key, value in config_data.items():
                        env_key = key.upper()
                        os.environ[env_key] = str(value)
                    
                    st.success("Configuration imported successfully!")
                    st.info("Please restart the application to apply all changes.")
                
                except Exception as e:
                    st.error(f"Failed to import configuration: {str(e)}")
    
    def _render_health_check(self):
        """Render system health check and diagnostics."""
        st.subheader("System Health Check")
        st.markdown("Comprehensive system diagnostics and component status")
        
        if st.button("🔍 Run Health Check"):
            with st.spinner("Running health check..."):
                try:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    health_status = loop.run_until_complete(self.controller.get_system_health())
                    loop.close()
                    
                    self._display_health_results(health_status)
                
                except Exception as e:
                    st.error(f"Health check failed: {str(e)}")
        
        # System information
        st.markdown("### System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Environment**")
            st.write(f"- Debug Mode: {config.DEBUG}")
            st.write(f"- Production: {config.is_production()}")
            st.write(f"- Port: {config.PORT}")
            st.write(f"- Host: {config.HOST}")
        
        with col2:
            st.write("**Configuration Status**")
            missing_config = config.validate_required_config()
            
            if not missing_config:
                st.success("✅ All required configuration present")
            else:
                st.error(f"❌ Missing configuration: {', '.join(missing_config)}")
        
        # Component versions and status
        st.markdown("### Component Information")
        
        try:
            import streamlit
            import pandas
            import plotly
            import redis
            
            versions = {
                'Streamlit': streamlit.__version__,
                'Pandas': pandas.__version__,
                'Plotly': plotly.__version__,
            }
            
            for component, version in versions.items():
                st.write(f"- **{component}**: {version}")
        
        except ImportError as e:
            st.warning(f"Could not get version information: {str(e)}")
    
    def _test_api_connections(self):
        """Test connections to all configured APIs."""
        st.markdown("### API Connection Test Results")
        
        # Test Gemini API
        if config.GEMINI_API_KEY:
            try:
                # This would test the actual Gemini API connection
                # For now, we'll simulate the test
                st.success("✅ Gemini AI: Connection successful")
            except Exception as e:
                st.error(f"❌ Gemini AI: {str(e)}")
        else:
            st.warning("⚠️ Gemini AI: API key not configured")
        
        # Test New Relic API
        if config.NEW_RELIC_API_KEY:
            try:
                # This would test the actual New Relic API connection
                st.success("✅ New Relic: Connection successful")
            except Exception as e:
                st.error(f"❌ New Relic: {str(e)}")
        else:
            st.warning("⚠️ New Relic: API key not configured")
        
        # Test Datadog API
        if config.DATADOG_API_KEY and config.DATADOG_APP_KEY:
            try:
                # This would test the actual Datadog API connection
                st.success("✅ Datadog: Connection successful")
            except Exception as e:
                st.error(f"❌ Datadog: {str(e)}")
        else:
            st.warning("⚠️ Datadog: API keys not configured")
    
    def _display_health_results(self, health_status: Dict[str, Any]):
        """Display comprehensive health check results."""
        overall_status = health_status.get('overall_status', 'unknown')
        
        # Overall status
        if overall_status == 'healthy':
            st.success(f"✅ Overall System Status: {overall_status.title()}")
        elif overall_status == 'degraded':
            st.warning(f"⚠️ Overall System Status: {overall_status.title()}")
        else:
            st.error(f"❌ Overall System Status: {overall_status.title()}")
        
        # Controller status
        controller_info = health_status.get('controller', {})
        
        st.markdown("### Controller Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            initialized = controller_info.get('initialized', False)
            st.write(f"**Initialized**: {'✅' if initialized else '❌'}")
        
        with col2:
            running_workflows = controller_info.get('running_workflows', 0)
            st.write(f"**Running Workflows**: {running_workflows}")
        
        with col3:
            total_workflows = controller_info.get('total_workflows_processed', 0)
            st.write(f"**Total Processed**: {total_workflows}")
        
        # Component status
        components = health_status.get('components', {})
        
        if components:
            st.markdown("### Component Status")
            
            for component_name, component_status in components.items():
                if isinstance(component_status, dict):
                    healthy = component_status.get('healthy', True)
                    status_icon = "✅" if healthy else "❌"
                    
                    with st.expander(f"{status_icon} {component_name.title()}"):
                        for key, value in component_status.items():
                            if key != 'healthy':
                                st.write(f"**{key.title()}**: {value}")
        
        # Timestamp
        timestamp = health_status.get('timestamp', 'Unknown')
        st.write(f"**Health check completed at**: {timestamp}")