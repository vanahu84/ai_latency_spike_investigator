"""
Main dashboard UI component for real-time latency monitoring.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio
import time
import random

from controller import LatencySpikeController, WorkflowStatus
from models import SpikeEvent, SeverityLevel, Metric


class DashboardUI:
    """Main dashboard UI for real-time latency monitoring and charts."""
    
    def __init__(self, controller: LatencySpikeController):
        self.controller = controller
    
    def render(self):
        """Render the main dashboard."""
        st.title("🔍 Latency Spike Dashboard")
        st.markdown("Real-time monitoring and analysis of API latency spikes")
        
        # Get system health
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            health_status = loop.run_until_complete(self.controller.get_system_health())
            loop.close()
        except Exception as e:
            st.error(f"Failed to get system health: {str(e)}")
            return
        
        # System overview metrics
        self._render_system_overview(health_status)
        
        # Real-time metrics and charts
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_latency_charts()
        
        with col2:
            self._render_active_incidents()
        
        # Real-time alerts section
        st.markdown("---")
        self._render_real_time_alerts()
        
        # Live status updates
        st.markdown("---")
        self._render_live_status_updates()
        
        # Recent activity
        st.markdown("---")
        self._render_recent_activity()
        
        # Recommendation analytics
        st.markdown("---")
        self._render_recommendation_analytics()
    
    def _render_system_overview(self, health_status: Dict[str, Any]):
        """Render system overview metrics."""
        st.subheader("System Overview")
        
        # Create metrics columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            overall_status = health_status.get('overall_status', 'unknown')
            status_color = {
                'healthy': 'normal',
                'degraded': 'off',
                'unhealthy': 'inverse',
                'error': 'inverse'
            }.get(overall_status, 'off')
            
            st.metric(
                label="System Status",
                value=overall_status.title(),
                delta=None
            )
        
        with col2:
            controller_info = health_status.get('controller', {})
            running_workflows = controller_info.get('running_workflows', 0)
            total_workflows = controller_info.get('total_workflows_processed', 0)
            
            st.metric(
                label="Active Workflows",
                value=running_workflows,
                delta=f"Total: {total_workflows}"
            )
        
        with col3:
            # Get spike detector status
            spike_detector = health_status.get('components', {}).get('spike_detector', {})
            active_spikes = spike_detector.get('active_spikes', 0)
            
            st.metric(
                label="Active Spikes",
                value=active_spikes,
                delta="Currently detected"
            )
        
        with col4:
            # Component health summary
            components = health_status.get('components', {})
            healthy_components = sum(1 for comp in components.values() 
                                   if isinstance(comp, dict) and comp.get('healthy', True))
            total_components = len(components)
            
            st.metric(
                label="Components",
                value=f"{healthy_components}/{total_components}",
                delta="Healthy"
            )
        
        # Component status details
        if st.expander("Component Details"):
            components = health_status.get('components', {})
            
            for component_name, component_status in components.items():
                if isinstance(component_status, dict):
                    status = "✅" if component_status.get('healthy', True) else "❌"
                    st.write(f"{status} **{component_name.title()}**")
                    
                    if 'error' in component_status and component_status['error']:
                        st.error(f"Error: {component_status['error']}")
                    
                    if 'response_time_ms' in component_status:
                        st.write(f"Response time: {component_status['response_time_ms']:.1f}ms")
    
    def _render_latency_charts(self):
        """Render real-time latency monitoring charts."""
        st.subheader("Real-time Latency Monitoring")
        
        # Add real-time update controls
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("🔄 Refresh Data"):
                st.session_state.last_chart_refresh = datetime.now()
                st.rerun()
            
            auto_refresh = st.checkbox("Auto-refresh", value=True, key="chart_auto_refresh")
            
            if auto_refresh:
                refresh_interval = st.selectbox(
                    "Refresh every:",
                    [10, 30, 60, 120],
                    index=1,
                    format_func=lambda x: f"{x}s",
                    key="chart_refresh_interval"
                )
                
                # Check if refresh is needed
                last_refresh = st.session_state.get('last_chart_refresh', datetime.now() - timedelta(seconds=refresh_interval))
                time_since_refresh = (datetime.now() - last_refresh).total_seconds()
                
                if time_since_refresh >= refresh_interval:
                    st.session_state.last_chart_refresh = datetime.now()
                    st.rerun()
        
        with col1:
            st.write("**Live Metrics Dashboard** - Updates every 30 seconds")
            
            # Add time range selector
            time_range = st.selectbox(
                "Time Range:",
                ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
                index=2,
                key="chart_time_range"
            )
        
        # Generate sample data for demonstration
        # In a real implementation, this would come from the controller/metrics
        time_range_hours = {"Last Hour": 1, "Last 6 Hours": 6, "Last 24 Hours": 24, "Last 7 Days": 168}
        hours = time_range_hours.get(time_range, 24)
        sample_data = self._generate_sample_latency_data(hours)
        
        # Main latency trend chart
        fig_trend = go.Figure()
        
        # Add baseline latency
        fig_trend.add_trace(go.Scatter(
            x=sample_data['timestamp'],
            y=sample_data['baseline_latency'],
            mode='lines',
            name='Baseline Latency',
            line=dict(color='blue', width=2)
        ))
        
        # Add current latency
        fig_trend.add_trace(go.Scatter(
            x=sample_data['timestamp'],
            y=sample_data['current_latency'],
            mode='lines+markers',
            name='Current Latency',
            line=dict(color='green', width=2),
            marker=dict(size=4)
        ))
        
        # Add spike threshold
        threshold = 1000  # ms
        fig_trend.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="Spike Threshold"
        )
        
        # Highlight spikes
        spike_points = sample_data[sample_data['current_latency'] > threshold]
        if not spike_points.empty:
            fig_trend.add_trace(go.Scatter(
                x=spike_points['timestamp'],
                y=spike_points['current_latency'],
                mode='markers',
                name='Detected Spikes',
                marker=dict(color='red', size=8, symbol='triangle-up')
            ))
        
        fig_trend.update_layout(
            title="API Latency Trends (Last 24 Hours)",
            xaxis_title="Time",
            yaxis_title="Latency (ms)",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Endpoint breakdown
        st.subheader("Latency by Endpoint")
        
        endpoint_data = self._generate_sample_endpoint_data()
        
        fig_endpoints = px.bar(
            endpoint_data,
            x='endpoint',
            y='avg_latency',
            color='status',
            color_discrete_map={
                'normal': 'green',
                'warning': 'orange',
                'critical': 'red'
            },
            title="Average Latency by Endpoint (Last Hour)"
        )
        
        fig_endpoints.update_layout(height=300)
        st.plotly_chart(fig_endpoints, use_container_width=True)
    
    def _render_active_incidents(self):
        """Render active incidents and alerts."""
        st.subheader("Active Incidents")
        
        # Get running workflows
        running_workflows = self.controller.get_running_workflows()
        
        if not running_workflows:
            st.info("No active incidents detected")
        else:
            for workflow_id, workflow_info in running_workflows.items():
                status = workflow_info['status']
                endpoint = workflow_info.get('spike_endpoint', 'Unknown')
                duration = workflow_info.get('duration_ms', 0) / 1000
                
                # Status indicator
                status_color = {
                    'detecting': '🔍',
                    'correlating': '🔗',
                    'analyzing': '🧠',
                    'enhancing': '✨',
                    'storing': '💾',
                    'completed': '✅',
                    'failed': '❌'
                }.get(status, '⏳')
                
                with st.container():
                    st.write(f"{status_color} **{endpoint}**")
                    st.write(f"Status: {status.title()}")
                    st.write(f"Duration: {duration:.1f}s")
                    
                    if st.button(f"View Details", key=f"view_{workflow_id}"):
                        st.session_state.selected_incident_id = workflow_id
                        st.session_state.navigation = "Incident Details"
                        st.rerun()
                    
                    st.markdown("---")
        
        # Recent incidents summary
        st.subheader("Recent Incidents")
        
        try:
            recent_incidents = self.controller.get_recent_incidents(limit=5)
            
            if not recent_incidents:
                st.info("No recent incidents")
            else:
                for incident in recent_incidents:
                    severity = incident.get('severity', 'unknown')
                    endpoint = incident.get('endpoint', 'Unknown')
                    timestamp = incident.get('timestamp', 'Unknown')
                    
                    severity_emoji = {
                        'low': '🟢',
                        'medium': '🟡',
                        'high': '🟠',
                        'critical': '🔴'
                    }.get(severity, '⚪')
                    
                    st.write(f"{severity_emoji} {endpoint} - {timestamp}")
        
        except Exception as e:
            st.error(f"Failed to load recent incidents: {str(e)}")
    
    def _render_recent_activity(self):
        """Render recent system activity and logs."""
        st.subheader("Recent Activity")
        
        # Activity tabs
        tab1, tab2, tab3 = st.tabs(["Workflow History", "System Events", "Performance Metrics"])
        
        with tab1:
            st.write("Recent workflow executions and their outcomes")
            
            # Sample workflow history
            workflow_history = [
                {
                    'timestamp': datetime.now() - timedelta(minutes=5),
                    'endpoint': '/api/users',
                    'status': 'completed',
                    'duration': '2.3s',
                    'causes_found': 2
                },
                {
                    'timestamp': datetime.now() - timedelta(minutes=15),
                    'endpoint': '/api/orders',
                    'status': 'completed',
                    'duration': '1.8s',
                    'causes_found': 1
                },
                {
                    'timestamp': datetime.now() - timedelta(minutes=32),
                    'endpoint': '/api/products',
                    'status': 'failed',
                    'duration': '0.5s',
                    'causes_found': 0
                }
            ]
            
            for workflow in workflow_history:
                status_emoji = '✅' if workflow['status'] == 'completed' else '❌'
                st.write(
                    f"{status_emoji} {workflow['timestamp'].strftime('%H:%M:%S')} - "
                    f"{workflow['endpoint']} ({workflow['duration']}) - "
                    f"{workflow['causes_found']} causes identified"
                )
        
        with tab2:
            st.write("System events and component status changes")
            
            # Sample system events
            system_events = [
                {
                    'timestamp': datetime.now() - timedelta(minutes=2),
                    'event': 'Spike detector threshold updated',
                    'level': 'info'
                },
                {
                    'timestamp': datetime.now() - timedelta(minutes=8),
                    'event': 'Gemini AI engine rate limit reached',
                    'level': 'warning'
                },
                {
                    'timestamp': datetime.now() - timedelta(minutes=12),
                    'event': 'Database connection restored',
                    'level': 'info'
                }
            ]
            
            for event in system_events:
                level_emoji = {'info': 'ℹ️', 'warning': '⚠️', 'error': '❌'}.get(event['level'], 'ℹ️')
                st.write(
                    f"{level_emoji} {event['timestamp'].strftime('%H:%M:%S')} - {event['event']}"
                )
        
        with tab3:
            st.write("System performance metrics and resource usage")
            
            # Performance metrics
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            
            with perf_col1:
                st.metric("Avg Response Time", "1.2s", "-0.3s")
            
            with perf_col2:
                st.metric("Memory Usage", "245 MB", "+12 MB")
            
            with perf_col3:
                st.metric("Cache Hit Rate", "87%", "+2%")
    
    def _generate_sample_latency_data(self, hours: int = 24) -> pd.DataFrame:
        """Generate sample latency data for demonstration."""
        now = datetime.now()
        # Generate data points based on time range
        if hours <= 1:
            # Every minute for last hour
            timestamps = [now - timedelta(minutes=60-i) for i in range(60)]
        elif hours <= 6:
            # Every 10 minutes for last 6 hours
            timestamps = [now - timedelta(minutes=(hours*60)-i*10) for i in range(hours*6)]
        elif hours <= 24:
            # Every hour for last 24 hours
            timestamps = [now - timedelta(hours=hours-i) for i in range(hours)]
        else:
            # Every 4 hours for longer periods
            timestamps = [now - timedelta(hours=hours-i*4) for i in range(hours//4)]
        
        # Generate realistic latency data with some spikes
        import random
        baseline = [200 + random.gauss(0, 50) for _ in range(len(timestamps))]
        current = []
        
        for i, base in enumerate(baseline):
            # Add some spikes randomly
            if random.random() < 0.1:  # 10% chance of spike
                spike_multiplier = random.uniform(3, 8)
                current.append(base * spike_multiplier)
            else:
                current.append(base + random.gauss(0, 30))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'baseline_latency': baseline,
            'current_latency': current
        })
    
    def _generate_sample_endpoint_data(self) -> pd.DataFrame:
        """Generate sample endpoint data for demonstration."""
        endpoints = ['/api/users', '/api/orders', '/api/products', '/api/auth', '/api/search']
        
        data = []
        for endpoint in endpoints:
            avg_latency = random.uniform(100, 2000)
            status = 'normal'
            if avg_latency > 1000:
                status = 'critical'
            elif avg_latency > 500:
                status = 'warning'
            
            data.append({
                'endpoint': endpoint,
                'avg_latency': avg_latency,
                'status': status
            })
        
        return pd.DataFrame(data)
    
    def _render_real_time_alerts(self):
        """Render real-time alerts and notifications."""
        st.subheader("🚨 Real-time Alerts")
        
        # Alert configuration
        col1, col2 = st.columns([3, 1])
        
        with col2:
            alert_enabled = st.checkbox("Enable Alerts", value=True, key="alerts_enabled")
            
            if alert_enabled:
                alert_threshold = st.selectbox(
                    "Alert Level:",
                    ["Critical Only", "High & Critical", "All Levels"],
                    index=1,
                    key="alert_threshold"
                )
        
        with col1:
            if alert_enabled:
                # Check for active alerts
                running_workflows = self.controller.get_running_workflows()
                active_alerts = []
                
                for workflow_id, workflow_info in running_workflows.items():
                    severity = workflow_info.get('severity', 'unknown')
                    endpoint = workflow_info.get('spike_endpoint', 'Unknown')
                    status = workflow_info.get('status', 'unknown')
                    
                    # Filter by alert threshold
                    if alert_threshold == "Critical Only" and severity != 'critical':
                        continue
                    elif alert_threshold == "High & Critical" and severity not in ['high', 'critical']:
                        continue
                    
                    active_alerts.append({
                        'id': workflow_id,
                        'severity': severity,
                        'endpoint': endpoint,
                        'status': status,
                        'timestamp': datetime.now()
                    })
                
                if active_alerts:
                    st.warning(f"⚠️ {len(active_alerts)} active alert(s)")
                    
                    for alert in active_alerts:
                        severity_emoji = {
                            'low': '🟢',
                            'medium': '🟡',
                            'high': '🟠',
                            'critical': '🔴'
                        }.get(alert['severity'], '⚪')
                        
                        alert_col1, alert_col2, alert_col3 = st.columns([1, 2, 1])
                        
                        with alert_col1:
                            st.write(f"{severity_emoji} {alert['severity'].upper()}")
                        
                        with alert_col2:
                            st.write(f"**{alert['endpoint']}** - {alert['status']}")
                        
                        with alert_col3:
                            if st.button("View", key=f"alert_view_{alert['id']}"):
                                st.session_state.selected_incident_id = alert['id']
                                st.session_state.navigation = "Incident Details"
                                st.rerun()
                else:
                    st.success("✅ No active alerts")
            else:
                st.info("Alerts disabled")
    
    def _render_live_status_updates(self):
        """Render live status updates and system activity."""
        st.subheader("📡 Live Status Updates")
        
        # Status update controls
        col1, col2 = st.columns([3, 1])
        
        with col2:
            live_updates = st.checkbox("Live Updates", value=True, key="live_updates_enabled")
            
            if live_updates:
                update_frequency = st.selectbox(
                    "Update every:",
                    [5, 10, 15, 30],
                    index=1,
                    format_func=lambda x: f"{x}s",
                    key="live_update_frequency"
                )
        
        with col1:
            if live_updates:
                # Create a container for live updates
                status_container = st.container()
                
                with status_container:
                    # Current system metrics
                    current_time = datetime.now()
                    
                    # Simulate live metrics (in real implementation, get from controller)
                    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                    
                    with metrics_col1:
                        # Active connections
                        active_connections = random.randint(45, 85)
                        st.metric("Active Connections", active_connections, delta=random.randint(-5, 5))
                    
                    with metrics_col2:
                        # Requests per minute
                        requests_per_min = random.randint(150, 300)
                        st.metric("Requests/min", requests_per_min, delta=random.randint(-20, 20))
                    
                    with metrics_col3:
                        # Average response time
                        avg_response = random.uniform(180, 250)
                        st.metric("Avg Response", f"{avg_response:.0f}ms", delta=f"{random.randint(-30, 30)}ms")
                    
                    with metrics_col4:
                        # Error rate
                        error_rate = random.uniform(0.1, 2.5)
                        st.metric("Error Rate", f"{error_rate:.1f}%", delta=f"{random.uniform(-0.5, 0.5):.1f}%")
                    
                    # Recent activity feed
                    st.write("**Recent Activity:**")
                    
                    # Generate recent activity (in real implementation, get from logs/events)
                    activities = [
                        f"{current_time.strftime('%H:%M:%S')} - Spike detection completed for /api/users",
                        f"{(current_time - timedelta(seconds=15)).strftime('%H:%M:%S')} - Correlation analysis started",
                        f"{(current_time - timedelta(seconds=32)).strftime('%H:%M:%S')} - New threshold configuration applied",
                        f"{(current_time - timedelta(seconds=45)).strftime('%H:%M:%S')} - Cache refresh completed",
                        f"{(current_time - timedelta(seconds=67)).strftime('%H:%M:%S')} - Gemini AI analysis completed"
                    ]
                    
                    for activity in activities[:3]:  # Show last 3 activities
                        st.text(activity)
                
                # Auto-refresh logic for live updates
                if live_updates:
                    last_live_update = st.session_state.get('last_live_update', datetime.now() - timedelta(seconds=update_frequency))
                    time_since_update = (datetime.now() - last_live_update).total_seconds()
                    
                    if time_since_update >= update_frequency:
                        st.session_state.last_live_update = datetime.now()
                        st.rerun()
            else:
                st.info("Live updates disabled")
    
    def _render_recommendation_analytics(self):
        """Render recommendation effectiveness analytics."""
        st.subheader("📊 Recommendation Analytics")
        
        try:
            # Get analytics data from controller
            analytics = self.controller.get_recommendation_analytics()
            
            if not analytics or not analytics.get('feedback_summary', {}).get('total_feedback', 0):
                st.info("No recommendation feedback data available yet")
                return
            
            feedback_summary = analytics.get('feedback_summary', {})
            action_completion = analytics.get('action_completion', {})
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_feedback = feedback_summary.get('total_feedback', 0)
                st.metric("Total Feedback", total_feedback)
            
            with col2:
                completion_rate = action_completion.get('completion_rate', 0)
                st.metric("Action Completion Rate", f"{completion_rate}%")
            
            with col3:
                very_helpful = feedback_summary.get('very_helpful', 0)
                helpful_rate = (very_helpful / total_feedback * 100) if total_feedback > 0 else 0
                st.metric("Very Helpful Rate", f"{helpful_rate:.1f}%")
            
            with col4:
                successful_impl = feedback_summary.get('successful_impl', 0)
                impl_rate = (successful_impl / total_feedback * 100) if total_feedback > 0 else 0
                st.metric("Success Implementation", f"{impl_rate:.1f}%")
            
            # Effectiveness breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Effectiveness Distribution**")
                effectiveness_data = {
                    'Very Helpful': feedback_summary.get('very_helpful', 0),
                    'Somewhat Helpful': feedback_summary.get('somewhat_helpful', 0),
                    'Not Helpful': feedback_summary.get('not_helpful', 0),
                    'Made it Worse': feedback_summary.get('made_worse', 0)
                }
                
                if sum(effectiveness_data.values()) > 0:
                    fig_effectiveness = px.pie(
                        values=list(effectiveness_data.values()),
                        names=list(effectiveness_data.keys()),
                        title="Recommendation Effectiveness",
                        color_discrete_map={
                            'Very Helpful': '#2E8B57',
                            'Somewhat Helpful': '#90EE90',
                            'Not Helpful': '#FFD700',
                            'Made it Worse': '#FF6347'
                        }
                    )
                    fig_effectiveness.update_layout(height=300)
                    st.plotly_chart(fig_effectiveness, use_container_width=True)
                else:
                    st.info("No effectiveness data available")
            
            with col2:
                st.write("**Implementation Time Distribution**")
                time_dist = analytics.get('implementation_time_distribution', [])
                
                if time_dist:
                    time_df = pd.DataFrame(time_dist)
                    fig_time = px.bar(
                        time_df,
                        x='time_to_implement',
                        y='count',
                        title="Time to Implement Recommendations",
                        labels={'time_to_implement': 'Time Range', 'count': 'Count'}
                    )
                    fig_time.update_layout(height=300)
                    st.plotly_chart(fig_time, use_container_width=True)
                else:
                    st.info("No implementation time data available")
            
            # Recent trends
            recent_trends = analytics.get('recent_trends', [])
            if recent_trends:
                st.write("**Recent Feedback Trends (Last 30 Days)**")
                
                trends_df = pd.DataFrame(recent_trends)
                if not trends_df.empty:
                    fig_trends = go.Figure()
                    
                    fig_trends.add_trace(go.Scatter(
                        x=trends_df['date'],
                        y=trends_df['feedback_count'],
                        mode='lines+markers',
                        name='Feedback Count',
                        yaxis='y'
                    ))
                    
                    fig_trends.add_trace(go.Scatter(
                        x=trends_df['date'],
                        y=trends_df['avg_effectiveness_score'],
                        mode='lines+markers',
                        name='Avg Effectiveness Score',
                        yaxis='y2',
                        line=dict(color='orange')
                    ))
                    
                    fig_trends.update_layout(
                        title="Feedback Trends Over Time",
                        xaxis_title="Date",
                        yaxis=dict(title="Feedback Count", side="left"),
                        yaxis2=dict(title="Effectiveness Score", side="right", overlaying="y"),
                        height=400
                    )
                    
                    st.plotly_chart(fig_trends, use_container_width=True)
        
        except Exception as e:
            st.error(f"Failed to load recommendation analytics: {str(e)}")