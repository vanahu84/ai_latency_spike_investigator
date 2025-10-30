# Patched Streamlit app with video embed

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


def start_health_check_server():
    try:
        from startup_checks import create_health_check_endpoint
        app = create_health_check_endpoint()
        if app:
            health_port = int(os.getenv('HEALTH_CHECK_PORT', '8502'))
            app.run(host='0.0.0.0', port=health_port, debug=False)
    except Exception:
        pass

if config.is_production():
    health_thread = threading.Thread(target=start_health_check_server, daemon=True)
    health_thread.start()


def main():
    st.set_page_config(
        page_title="Latency Spike Investigator",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ---- Embed Video at Top ----
    st.markdown("### 📺 Watch Demo")
    video_url = "https://drive.google.com/file/d/1vDIZzPJ8fHm7H8qkBQ_c3RlZk_NJBl89/preview"
    st.components.v1.iframe(video_url, height=380)
    st.markdown("---")

    if 'controller' not in st.session_state:
        st.session_state.controller = None
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    if 'selected_incident_id' not in st.session_state:
        st.session_state.selected_incident_id = None

    st.sidebar.title("🔍 Latency Spike Investigator")
    st.sidebar.markdown("---")

    page = st.sidebar.selectbox(
        "Navigate to:",
        ["Dashboard", "Incident Details", "Historical Incidents", "Configuration"],
        key="navigation"
    )

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

    if st.sidebar.button("🔄 Refresh Now"):
        st.session_state.last_refresh = datetime.now()
        st.rerun()

    st.sidebar.markdown("### System Status")

    if st.session_state.controller is None:
        with st.sidebar:
            with st.spinner("Initializing system..."):
                try:
                    controller = LatencySpikeController()
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

    st.sidebar.text(f"Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

    if page == "Dashboard":
        DashboardUI(st.session_state.controller).render()
    elif page == "Incident Details":
        IncidentDetailUI(st.session_state.controller).render()
    elif page == "Historical Incidents":
        HistoricalIncidentsUI(st.session_state.controller).render()
    elif page == "Configuration":
        ConfigurationUI(st.session_state.controller).render()

    if auto_refresh and st.session_state.controller:
        time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
        if time_since_refresh >= refresh_interval:
            st.session_state.last_refresh = datetime.now()
            st.rerun()


if __name__ == "__main__":
    main()
