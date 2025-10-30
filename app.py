"""
Main entry point for the Latency Spike Investigator application.
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time
import json
import threading
import os

from config import config
from controller import LatencySpikeController, WorkflowResult, WorkflowStatus
from models import SpikeEvent, SeverityLevel, Metric
from ui.dashboard import DashboardUI
from ui.incidents import IncidentDetailUI, HistoricalIncidentsUI
from ui.configuration import ConfigurationUI


# Health check endpoint for deployment platforms
def start_health_check_server():
    """Start a simple health check server in a separate thread."""
    try:
        from startup_checks import create_health_check_endpoint
        
        app = create_health_check_endpoint()
        if app:
            # Run on a different port to avoid conflicts with Streamlit
            health_port = int(os.getenv('HEALTH_CHECK_PORT', '8502'))
            app.run(host='0.0.0.0', port=health_port, debug=False)
    except Exception as e:
        # Health check is optional - don't fail the main app
        pass


# Start health check server in background (optional)
if config.is_production():
    health_thread = threading.Thread(target=start_health_check_server, daemon=True)
    health_thread.start()


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Latency Spike Investigator",
        # ---- Embed Video at Top ----
        st.markdown("### 📺 Watch Demo")
        video_url = "https://drive.google.com/file/d/1vDIZzPJ8fHm7H8qkBQ_c3RlZk_NJBl89/preview"
        st.components.v1.iframe(video_url, height=380)
        st.markdown("---"),

        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'controller' not in st.session_state:
        st.session_state.controller = None
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    if 'selected_incident_id' not in st.session_state:
        st.session_state.selected_incident_id = None
    
    # Sidebar navigation
    st.sidebar.title("🔍 Latency Spike Investigator")
    st.sidebar.markdown("---")
    
    # Navigation menu
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["Dashboard", "Incident Details", "Historical Incidents", "Configuration"],
        key="navigation"
    )
    
    # Auto-refresh controls
    st.sidebar.markdown("### Auto-Refresh")
    auto_refresh = st.sidebar.checkbox("Enable auto-refresh", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh
    
    if auto_refresh:
        refresh_interval = st.sidebar.selectbox(
            "Refresh interval:",
            [5, 10, 30, 60],
            index=1,
            format_func=lambda x: f"{x} seconds"
        )
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.session_state.last_refresh = datetime.now()
        st.rerun()
    
    # System status indicator
    st.sidebar.markdown("### System Status")
    
    # Initialize controller if needed
    if st.session_state.controller is None:
        with st.sidebar:
            with st.spinner("Initializing system..."):
                try:
                    controller = LatencySpikeController()
                    # Run async initialization
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    success = loop.run_until_complete(controller.initialize())
                    loop.close()
                    
                    if success:
                        st.session_state.controller = controller
                        st.sidebar.success("✅ System initialized")
                    else:
                        st.sidebar.error("❌ Initialization failed")
                        st.error("Failed to initialize the system. Please check your configuration.")
                        return
                except Exception as e:
                    st.sidebar.error(f"❌ Error: {str(e)}")
                    st.error(f"System initialization error: {str(e)}")
                    return
    else:
        st.sidebar.success("✅ System running")
    
    # Display last refresh time
    st.sidebar.text(f"Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    
    # Route to appropriate page
    if page == "Dashboard":
        dashboard_ui = DashboardUI(st.session_state.controller)
        dashboard_ui.render()
    elif page == "Incident Details":
        incident_ui = IncidentDetailUI(st.session_state.controller)
        incident_ui.render()
    elif page == "Historical Incidents":
        historical_ui = HistoricalIncidentsUI(st.session_state.controller)
        historical_ui.render()
    elif page == "Configuration":
        config_ui = ConfigurationUI(st.session_state.controller)
        config_ui.render()
    
    # Auto-refresh logic
    if auto_refresh and st.session_state.controller:
        time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
        if time_since_refresh >= refresh_interval:
            st.session_state.last_refresh = datetime.now()
            st.rerun()


if __name__ == "__main__":

    main()

