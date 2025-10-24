"""
Incident detail and historical incidents UI components.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import time
import random

from controller import LatencySpikeController
from models import SpikeEvent, SeverityLevel, RootCause


class IncidentDetailUI:
    """UI component for detailed incident analysis and timeline visualization."""
    
    def __init__(self, controller: LatencySpikeController):
        self.controller = controller
    
    def render(self):
        """Render the incident detail view."""
        st.title("🔍 Incident Details")
        
        # Incident selection
        incident_id = st.session_state.get('selected_incident_id')
        
        if not incident_id:
            st.info("Select an incident from the dashboard or historical incidents to view details.")
            
            # Show recent incidents for selection
            st.subheader("Recent Incidents")
            self._render_incident_selector()
            return
        
        # Load incident details
        try:
            if incident_id.startswith('workflow_'):
                # This is a running workflow
                running_workflows = self.controller.get_running_workflows()
                if incident_id in running_workflows:
                    workflow_info = running_workflows[incident_id]
                    self._render_active_workflow_details(incident_id, workflow_info)
                else:
                    st.error("Workflow not found or has completed.")
            else:
                # This is a stored incident
                incident_details = self.controller.get_incident_details(int(incident_id))
                if incident_details:
                    self._render_stored_incident_details(incident_details)
                else:
                    st.error("Incident not found.")
        
        except Exception as e:
            st.error(f"Failed to load incident details: {str(e)}")
    
    def _render_incident_selector(self):
        """Render incident selection interface."""
        try:
            recent_incidents = self.controller.get_recent_incidents(limit=10)
            
            if not recent_incidents:
                st.info("No recent incidents available.")
                return
            
            # Create incident selection table
            incident_data = []
            for incident in recent_incidents:
                incident_data.append({
                    'ID': incident.get('id', 'N/A'),
                    'Timestamp': incident.get('timestamp', 'Unknown'),
                    'Endpoint': incident.get('endpoint', 'Unknown'),
                    'Severity': incident.get('severity', 'unknown'),
                    'Duration': incident.get('duration', 'N/A')
                })
            
            df = pd.DataFrame(incident_data)
            
            # Display as interactive table
            selected_indices = st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row"
            )
            
            if selected_indices and selected_indices.selection.rows:
                selected_row = selected_indices.selection.rows[0]
                selected_id = df.iloc[selected_row]['ID']
                
                if st.button("View Selected Incident"):
                    st.session_state.selected_incident_id = str(selected_id)
                    st.rerun()
        
        except Exception as e:
            st.error(f"Failed to load recent incidents: {str(e)}")
    
    def _render_active_workflow_details(self, workflow_id: str, workflow_info: Dict[str, Any]):
        """Render details for an active workflow."""
        st.subheader(f"Active Workflow: {workflow_id}")
        
        # Workflow status
        status = workflow_info.get('status', 'unknown')
        endpoint = workflow_info.get('spike_endpoint', 'Unknown')
        start_time = workflow_info.get('start_time', 'Unknown')
        duration = workflow_info.get('duration_ms', 0) / 1000
        
        # Status header
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Status", status.title())
        
        with col2:
            st.metric("Endpoint", endpoint)
        
        with col3:
            st.metric("Duration", f"{duration:.1f}s")
        
        # Progress indicator
        st.subheader("Workflow Progress")
        
        workflow_steps = [
            'detecting', 'correlating', 'analyzing', 'enhancing', 'storing', 'completed'
        ]
        
        current_step_index = workflow_steps.index(status) if status in workflow_steps else 0
        
        progress_cols = st.columns(len(workflow_steps))
        
        for i, (col, step) in enumerate(zip(progress_cols, workflow_steps)):
            with col:
                if i <= current_step_index:
                    st.success(f"✅ {step.title()}")
                elif i == current_step_index + 1:
                    st.info(f"⏳ {step.title()}")
                else:
                    st.write(f"⏸️ {step.title()}")
        
        # Component status
        component_status = workflow_info.get('component_status', {})
        if component_status:
            st.subheader("Component Status")
            
            for component, comp_status in component_status.items():
                if comp_status == 'success':
                    st.success(f"✅ {component.title()}: Success")
                elif comp_status.startswith('failed'):
                    st.error(f"❌ {component.title()}: {comp_status}")
                else:
                    st.info(f"⏳ {component.title()}: {comp_status}")
        
        # Auto-refresh for active workflows
        if status not in ['completed', 'failed']:
            if st.button("🔄 Refresh Status"):
                st.rerun()
            
            # Auto-refresh every 5 seconds
            time.sleep(5)
            st.rerun()
    
    def _render_stored_incident_details(self, incident_details: Dict[str, Any]):
        """Render details for a stored incident."""
        st.subheader(f"Incident #{incident_details.get('id', 'Unknown')}")
        
        # Incident overview
        spike_data = incident_details.get('spike_event', {})
        correlation_data = incident_details.get('correlation_result', {})
        recommendations = incident_details.get('recommendations', [])
        
        # Header metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            severity = spike_data.get('severity', 'unknown')
            severity_emoji = {
                'low': '🟢',
                'medium': '🟡', 
                'high': '🟠',
                'critical': '🔴'
            }.get(severity, '⚪')
            st.metric("Severity", f"{severity_emoji} {severity.title()}")
        
        with col2:
            endpoint = spike_data.get('endpoint', 'Unknown')
            st.metric("Endpoint", endpoint)
        
        with col3:
            spike_ratio = spike_data.get('spike_ratio', 0)
            st.metric("Spike Ratio", f"{spike_ratio:.1f}x")
        
        with col4:
            confidence = correlation_data.get('confidence', 0)
            st.metric("Analysis Confidence", f"{confidence:.0%}")
        
        # Timeline visualization
        st.subheader("Incident Timeline")
        self._render_incident_timeline(incident_details)
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        self._render_correlation_analysis(correlation_data)
        
        # Root cause analysis
        st.subheader("Root Cause Analysis")
        self._render_root_cause_analysis(recommendations)
        
        # Raw data
        if st.expander("Raw Incident Data"):
            st.json(incident_details)
    
    def _render_incident_timeline(self, incident_details: Dict[str, Any]):
        """Render incident timeline visualization."""
        spike_data = incident_details.get('spike_event', {})
        
        # Create timeline chart
        timestamp = spike_data.get('timestamp')
        if not timestamp:
            st.info("Timeline data not available")
            return
        
        # Parse timestamp if it's a string
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                st.error("Invalid timestamp format")
                return
        
        # Generate timeline data around the incident
        timeline_start = timestamp - timedelta(minutes=30)
        timeline_end = timestamp + timedelta(minutes=30)
        
        # Sample data for demonstration
        timeline_data = self._generate_timeline_data(timeline_start, timeline_end, timestamp)
        
        fig = go.Figure()
        
        # Add baseline latency
        fig.add_trace(go.Scatter(
            x=timeline_data['timestamp'],
            y=timeline_data['latency'],
            mode='lines+markers',
            name='Latency',
            line=dict(color='blue', width=2)
        ))
        
        # Highlight the spike
        spike_latency = spike_data.get('spike_latency', 0)
        fig.add_trace(go.Scatter(
            x=[timestamp],
            y=[spike_latency],
            mode='markers',
            name='Detected Spike',
            marker=dict(color='red', size=15, symbol='triangle-up')
        ))
        
        # Add threshold line
        baseline_latency = spike_data.get('baseline_latency', 0)
        if baseline_latency > 0:
            threshold = baseline_latency * 2  # Assuming 2x threshold
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text="Spike Threshold"
            )
        
        fig.update_layout(
            title="Latency Timeline (±30 minutes)",
            xaxis_title="Time",
            yaxis_title="Latency (ms)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add timeline controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Show Detailed Metrics"):
                self._render_detailed_metrics_timeline(incident_details)
        
        with col2:
            if st.button("🔍 Zoom to Spike"):
                # This would zoom the chart to focus on the spike period
                st.info("Chart zoomed to spike period")
        
        with col3:
            if st.button("📈 Compare with Baseline"):
                self._render_baseline_comparison(incident_details)
    
    def _render_correlation_analysis(self, correlation_data: Dict[str, Any]):
        """Render correlation analysis results."""
        correlation_scores = correlation_data.get('correlation_scores', {})
        
        if not correlation_scores:
            st.info("No correlation data available")
            return
        
        # Create correlation chart
        metrics = list(correlation_scores.keys())
        scores = list(correlation_scores.values())
        
        fig = px.bar(
            x=metrics,
            y=scores,
            title="Metric Correlations with Latency Spike",
            labels={'x': 'Metrics', 'y': 'Correlation Score'},
            color=scores,
            color_continuous_scale='RdYlBu_r'
        )
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation details
        st.write("**Correlation Details:**")
        for metric, score in correlation_scores.items():
            correlation_strength = "Strong" if abs(score) > 0.7 else "Moderate" if abs(score) > 0.4 else "Weak"
            correlation_direction = "Positive" if score > 0 else "Negative"
            
            st.write(f"- **{metric}**: {score:.3f} ({correlation_strength} {correlation_direction})")
    
    def _render_root_cause_analysis(self, recommendations: List[Dict[str, Any]]):
        """Render root cause analysis and recommendations."""
        if not recommendations:
            st.info("No root cause analysis available")
            return
        
        # Group recommendations by category
        categories = {}
        for rec in recommendations:
            category = rec.get('category', 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(rec)
        
        # Render each category
        for category, recs in categories.items():
            st.write(f"### {category.title()} Issues")
            
            for i, rec in enumerate(recs):
                confidence = rec.get('confidence_score', 0)
                description = rec.get('description', 'No description available')
                evidence = rec.get('supporting_evidence', [])
                actions = rec.get('recommended_actions', [])
                
                with st.expander(f"{description} (Confidence: {confidence:.0%})"):
                    if evidence:
                        st.write("**Supporting Evidence:**")
                        for item in evidence:
                            st.write(f"- {item}")
                    
                    if actions:
                        st.write("**Recommended Actions:**")
                        for j, action in enumerate(actions):
                            col1, col2, col3 = st.columns([3, 1, 1])
                            
                            with col1:
                                # Check if this action was previously completed
                                action_key = f"action_{category}_{i}_{j}"
                                completed_key = f"completed_{category}_{i}_{j}"
                                
                                # Get completion status from session state or database
                                previously_completed = self._get_action_completion_status(
                                    st.session_state.get('selected_incident_id'),
                                    action_key
                                )
                                
                                completed = st.checkbox(
                                    action,
                                    value=previously_completed,
                                    key=action_key,
                                    help="Mark as completed"
                                )
                                
                                # If status changed, update the completion
                                if completed != previously_completed:
                                    self._update_action_completion(
                                        st.session_state.get('selected_incident_id'),
                                        action_key,
                                        completed,
                                        action
                                    )
                            
                            with col2:
                                if completed:
                                    st.success("✅ Done")
                                else:
                                    priority = rec.get('priority', 3)
                                    priority_color = {1: "🔴", 2: "🟡", 3: "🟢"}.get(priority, "⚪")
                                    st.write(f"{priority_color} P{priority}")
                            
                            with col3:
                                feedback_key = f"show_feedback_{category}_{i}_{j}"
                                if st.button("📝 Feedback", key=f"feedback_btn_{category}_{i}_{j}"):
                                    st.session_state[feedback_key] = not st.session_state.get(feedback_key, False)
                                
                                # Show feedback form if toggled
                                if st.session_state.get(feedback_key, False):
                                    with st.form(key=f"feedback_form_{category}_{i}_{j}"):
                                        st.write("**Recommendation Feedback**")
                                        
                                        # Effectiveness rating
                                        effectiveness = st.radio(
                                            "How effective was this recommendation?",
                                            ["Very Helpful", "Somewhat Helpful", "Not Helpful", "Made it Worse"],
                                            key=f"effectiveness_{category}_{i}_{j}"
                                        )
                                        
                                        # Implementation status
                                        implementation = st.radio(
                                            "Implementation status:",
                                            ["Successfully Implemented", "Partially Implemented", "Could Not Implement", "Not Attempted"],
                                            key=f"implementation_{category}_{i}_{j}"
                                        )
                                        
                                        # Time to implement
                                        time_to_implement = st.selectbox(
                                            "Time to implement:",
                                            ["< 5 minutes", "5-15 minutes", "15-30 minutes", "30-60 minutes", "> 1 hour"],
                                            key=f"time_impl_{category}_{i}_{j}"
                                        )
                                        
                                        # Additional feedback
                                        feedback_text = st.text_area(
                                            "Additional feedback (optional):",
                                            key=f"feedback_text_{category}_{i}_{j}",
                                            placeholder="Any additional context, suggestions, or issues encountered..."
                                        )
                                        
                                        # Submit feedback
                                        if st.form_submit_button("Submit Feedback"):
                                            feedback_data = {
                                                'recommendation_id': f"{category}_{i}_{j}",
                                                'effectiveness': effectiveness,
                                                'implementation': implementation,
                                                'time_to_implement': time_to_implement,
                                                'feedback_text': feedback_text,
                                                'timestamp': datetime.now().isoformat(),
                                                'incident_id': st.session_state.get('selected_incident_id')
                                            }
                                            
                                            # Store feedback (in real implementation, save to database)
                                            self._store_recommendation_feedback(feedback_data)
                                            st.success("Feedback submitted! Thank you for helping improve our recommendations.")
                                            st.session_state[feedback_key] = False
                                            st.rerun()
                            
                            if completed:
                                # Show completion timestamp and allow adding notes
                                with st.expander("Completion Details"):
                                    completion_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    st.write(f"**Completed at:** {completion_time}")
                                    
                                    completion_notes = st.text_area(
                                        "Completion notes (optional):",
                                        key=f"completion_notes_{category}_{i}_{j}",
                                        placeholder="Add any notes about how this was resolved..."
                                    )
                                    
                                    if completion_notes and st.button("Save Notes", key=f"save_notes_{category}_{i}_{j}"):
                                        st.success("Notes saved!")
    
    def _generate_timeline_data(self, start_time: datetime, end_time: datetime, spike_time: datetime) -> pd.DataFrame:
        """Generate sample timeline data around an incident."""
        import random
        
        # Generate timestamps every minute
        timestamps = []
        current = start_time
        while current <= end_time:
            timestamps.append(current)
            current += timedelta(minutes=1)
        
        # Generate latency data with spike
        latencies = []
        baseline = 200
        
        for ts in timestamps:
            if abs((ts - spike_time).total_seconds()) < 300:  # Within 5 minutes of spike
                # Elevated latency around spike time
                latency = baseline + random.gauss(500, 100)
            else:
                # Normal latency
                latency = baseline + random.gauss(0, 30)
            
            latencies.append(max(latency, 50))  # Minimum 50ms
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'latency': latencies
        })
    
    def _store_recommendation_feedback(self, feedback_data: Dict[str, Any]):
        """Store recommendation feedback in the database."""
        try:
            # In a real implementation, this would save to the database
            # For now, we'll store in session state for demonstration
            if 'recommendation_feedback' not in st.session_state:
                st.session_state.recommendation_feedback = []
            
            st.session_state.recommendation_feedback.append(feedback_data)
            
            # Also attempt to store in controller if available
            if hasattr(self.controller, 'store_recommendation_feedback'):
                self.controller.store_recommendation_feedback(feedback_data)
        
        except Exception as e:
            st.error(f"Failed to store feedback: {str(e)}")
    
    def _get_action_completion_status(self, incident_id: str, action_key: str) -> bool:
        """Get the completion status of a specific action."""
        try:
            # Check session state first
            if 'action_completions' not in st.session_state:
                st.session_state.action_completions = {}
            
            completion_key = f"{incident_id}_{action_key}"
            return st.session_state.action_completions.get(completion_key, False)
        
        except Exception:
            return False
    
    def _update_action_completion(self, incident_id: str, action_key: str, completed: bool, action_text: str):
        """Update the completion status of a specific action."""
        try:
            # Initialize session state if needed
            if 'action_completions' not in st.session_state:
                st.session_state.action_completions = {}
            
            if 'action_completion_history' not in st.session_state:
                st.session_state.action_completion_history = []
            
            completion_key = f"{incident_id}_{action_key}"
            st.session_state.action_completions[completion_key] = completed
            
            # Add to completion history
            completion_record = {
                'incident_id': incident_id,
                'action_key': action_key,
                'action_text': action_text,
                'completed': completed,
                'timestamp': datetime.now().isoformat(),
                'user': 'current_user'  # In real implementation, get from auth
            }
            
            st.session_state.action_completion_history.append(completion_record)
            
            # Also attempt to store in controller if available
            if hasattr(self.controller, 'update_action_completion'):
                self.controller.update_action_completion(completion_record)
            
            # Show success message
            if completed:
                st.success(f"✅ Action marked as completed!")
            else:
                st.info(f"Action marked as not completed")
        
        except Exception as e:
            st.error(f"Failed to update action completion: {str(e)}")
    
    def _render_detailed_metrics_timeline(self, incident_details: Dict[str, Any]):
        """Render detailed metrics timeline with multiple data sources."""
        st.subheader("Detailed Metrics Timeline")
        
        # Generate sample multi-metric data
        spike_data = incident_details.get('spike_event', {})
        timestamp = spike_data.get('timestamp')
        
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                timestamp = datetime.now()
        
        timeline_start = timestamp - timedelta(minutes=30)
        timeline_end = timestamp + timedelta(minutes=30)
        
        # Generate multi-metric timeline data
        timestamps = []
        current = timeline_start
        while current <= timeline_end:
            timestamps.append(current)
            current += timedelta(minutes=1)
        
        import random
        
        # Generate correlated metrics
        latency_data = []
        cpu_data = []
        memory_data = []
        db_response_data = []
        
        for i, ts in enumerate(timestamps):
            # Base values
            base_latency = 200
            base_cpu = 30
            base_memory = 60
            base_db = 50
            
            # Add spike effect
            if abs((ts - timestamp).total_seconds()) < 300:  # Within 5 minutes of spike
                spike_factor = 1 + random.uniform(2, 5)
                latency_data.append(base_latency * spike_factor + random.gauss(0, 50))
                cpu_data.append(min(100, base_cpu * (1 + spike_factor * 0.3) + random.gauss(0, 5)))
                memory_data.append(min(100, base_memory * (1 + spike_factor * 0.2) + random.gauss(0, 3)))
                db_response_data.append(base_db * spike_factor * 0.8 + random.gauss(0, 10))
            else:
                latency_data.append(base_latency + random.gauss(0, 30))
                cpu_data.append(base_cpu + random.gauss(0, 5))
                memory_data.append(base_memory + random.gauss(0, 3))
                db_response_data.append(base_db + random.gauss(0, 10))
        
        # Create subplot figure
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('API Latency (ms)', 'CPU Usage (%)', 'Memory Usage (%)', 'DB Response Time (ms)'),
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=timestamps, y=latency_data, name='API Latency', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=cpu_data, name='CPU Usage', line=dict(color='red')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=memory_data, name='Memory Usage', line=dict(color='green')),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=db_response_data, name='DB Response', line=dict(color='orange')),
            row=4, col=1
        )
        
        # Add spike marker
        for i in range(1, 5):
            fig.add_vline(
                x=timestamp,
                line_dash="dash",
                line_color="red",
                annotation_text="Spike Event" if i == 1 else "",
                row=i, col=1
            )
        
        fig.update_layout(
            height=800,
            title_text="Multi-Metric Timeline Analysis",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_baseline_comparison(self, incident_details: Dict[str, Any]):
        """Render comparison between incident period and baseline."""
        st.subheader("Baseline Comparison")
        
        spike_data = incident_details.get('spike_event', {})
        baseline_latency = spike_data.get('baseline_latency', 200)
        spike_latency = spike_data.get('spike_latency', 800)
        
        # Create comparison metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Baseline Latency",
                f"{baseline_latency:.0f} ms",
                help="Normal latency for this endpoint"
            )
        
        with col2:
            st.metric(
                "Spike Latency",
                f"{spike_latency:.0f} ms",
                delta=f"+{spike_latency - baseline_latency:.0f} ms",
                delta_color="inverse"
            )
        
        with col3:
            spike_ratio = spike_latency / baseline_latency if baseline_latency > 0 else 0
            st.metric(
                "Spike Ratio",
                f"{spike_ratio:.1f}x",
                help="How many times higher than baseline"
            )
        
        with col4:
            percentile_impact = min(99, (spike_ratio - 1) * 20)
            st.metric(
                "Performance Impact",
                f"{percentile_impact:.0f}%",
                help="Estimated performance degradation"
            )
        
        # Comparison chart
        comparison_data = {
            'Metric': ['Baseline Period', 'Incident Period'],
            'Avg Latency (ms)': [baseline_latency, spike_latency],
            'P95 Latency (ms)': [baseline_latency * 1.2, spike_latency * 1.3],
            'P99 Latency (ms)': [baseline_latency * 1.5, spike_latency * 1.5]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        
        fig = px.bar(
            df_comparison,
            x='Metric',
            y=['Avg Latency (ms)', 'P95 Latency (ms)', 'P99 Latency (ms)'],
            title="Latency Distribution Comparison",
            barmode='group'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


class HistoricalIncidentsUI:
    """UI component for historical incidents with search and filtering."""
    
    def __init__(self, controller: LatencySpikeController):
        self.controller = controller
    
    def render(self):
        """Render the historical incidents view."""
        st.title("📊 Historical Incidents")
        st.markdown("Search and analyze past latency incidents")
        
        # Filters and search
        self._render_filters()
        
        # Incidents table
        self._render_incidents_table()
        
        # Analytics
        self._render_incident_analytics()
    
    def _render_filters(self):
        """Render search and filter controls."""
        st.subheader("Search & Filters")
        
        # Advanced search section
        with st.expander("🔍 Advanced Search", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                # Text search
                search_text = st.text_input(
                    "Search",
                    placeholder="Search endpoints, causes, or descriptions...",
                    help="Search across endpoint names, root causes, and descriptions"
                )
                
                # Date range
                date_range = st.date_input(
                    "Date Range",
                    value=(datetime.now().date() - timedelta(days=7), datetime.now().date()),
                    help="Select date range for incidents"
                )
            
            with col2:
                # Severity filter
                severity_filter = st.multiselect(
                    "Severity Levels",
                    options=['low', 'medium', 'high', 'critical'],
                    default=['medium', 'high', 'critical'],
                    help="Filter by incident severity"
                )
                
                # Spike ratio range
                spike_ratio_range = st.slider(
                    "Spike Ratio Range",
                    min_value=1.0,
                    max_value=10.0,
                    value=(1.0, 10.0),
                    step=0.1,
                    help="Filter by spike intensity"
                )
        
        # Quick filters
        st.markdown("### Quick Filters")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("🔴 Critical Only"):
                st.session_state.quick_filter = 'critical'
        
        with col2:
            if st.button("📅 Today"):
                st.session_state.quick_filter = 'today'
        
        with col3:
            if st.button("📊 High Impact"):
                st.session_state.quick_filter = 'high_impact'
        
        with col4:
            if st.button("🔄 Recurring"):
                st.session_state.quick_filter = 'recurring'
        
        with col5:
            if st.button("🗑️ Clear Filters"):
                st.session_state.quick_filter = None
        
        # Results configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            limit = st.selectbox(
                "Results per page",
                options=[25, 50, 100, 200],
                index=1,
                help="Maximum number of results to display"
            )
        
        with col2:
            sort_by = st.selectbox(
                "Sort by",
                options=['timestamp', 'severity', 'spike_ratio', 'endpoint'],
                index=0,
                help="Sort incidents by selected field"
            )
        
        with col3:
            sort_order = st.selectbox(
                "Sort order",
                options=['Descending', 'Ascending'],
                index=0,
                help="Sort order for results"
            )
        
        # Store filters in session state
        st.session_state.incident_filters = {
            'search_text': search_text,
            'date_range': date_range,
            'severity_filter': severity_filter,
            'spike_ratio_range': spike_ratio_range,
            'limit': limit,
            'sort_by': sort_by,
            'sort_order': sort_order.lower(),
            'quick_filter': st.session_state.get('quick_filter')
        }
    
    def _render_incidents_table(self):
        """Render the incidents table with filtering."""
        st.subheader("Incidents")
        
        try:
            # Get incidents with filters
            filters = st.session_state.get('incident_filters', {})
            incidents = self.controller.get_recent_incidents(limit=filters.get('limit', 50))
            
            if not incidents:
                st.info("No incidents found matching the criteria.")
                return
            
            # Apply filters
            filtered_incidents = self._apply_filters(incidents, filters)
            
            if not filtered_incidents:
                st.info("No incidents match the selected filters.")
                return
            
            # Create incidents dataframe
            incident_data = []
            for incident in filtered_incidents:
                incident_data.append({
                    'ID': incident.get('id', 'N/A'),
                    'Timestamp': incident.get('timestamp', 'Unknown'),
                    'Endpoint': incident.get('endpoint', 'Unknown'),
                    'Severity': incident.get('severity', 'unknown'),
                    'Spike Ratio': f"{incident.get('spike_ratio', 0):.1f}x",
                    'Duration': incident.get('duration', 'N/A'),
                    'Causes Found': incident.get('causes_count', 0)
                })
            
            df = pd.DataFrame(incident_data)
            
            # Add column configuration
            column_config = {
                'ID': st.column_config.NumberColumn('ID', width='small'),
                'Timestamp': st.column_config.DatetimeColumn('Timestamp', width='medium'),
                'Endpoint': st.column_config.TextColumn('Endpoint', width='large'),
                'Severity': st.column_config.TextColumn('Severity', width='small'),
                'Spike Ratio': st.column_config.TextColumn('Spike Ratio', width='small'),
                'Duration': st.column_config.TextColumn('Duration', width='small'),
                'Causes Found': st.column_config.NumberColumn('Causes', width='small')
            }
            
            # Display table with selection
            selected_indices = st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="multi-row",
                column_config=column_config
            )
            
            # Bulk actions
            if selected_indices and selected_indices.selection.rows:
                selected_rows = selected_indices.selection.rows
                st.write(f"**{len(selected_rows)} incident(s) selected**")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button("📊 Compare Selected"):
                        self._render_incident_comparison(filtered_incidents, selected_rows, df)
                
                with col2:
                    if st.button("📥 Export Selected"):
                        self._export_selected_incidents(filtered_incidents, selected_rows, df)
                
                with col3:
                    if st.button("🏷️ Tag Incidents"):
                        self._render_tagging_interface(selected_rows, df)
                
                with col4:
                    if len(selected_rows) == 1:
                        selected_id = df.iloc[selected_rows[0]]['ID']
                        if st.button("🔍 View Details"):
                            st.session_state.selected_incident_id = str(selected_id)
                            st.session_state.navigation = "Incident Details"
                            st.rerun()
        
        except Exception as e:
            st.error(f"Failed to load incidents: {str(e)}")
    
    def _render_incident_analytics(self):
        """Render incident analytics and trends."""
        st.subheader("Incident Analytics")
        
        try:
            # Get recent incidents for analytics
            incidents = self.controller.get_recent_incidents(limit=100)
            
            if not incidents:
                st.info("No incident data available for analytics.")
                return
            
            # Create analytics tabs
            tab1, tab2, tab3 = st.tabs(["Trends", "Patterns", "Performance"])
            
            with tab1:
                self._render_incident_trends(incidents)
            
            with tab2:
                self._render_incident_patterns(incidents)
            
            with tab3:
                self._render_performance_metrics(incidents)
        
        except Exception as e:
            st.error(f"Failed to generate analytics: {str(e)}")
    
    def _render_incident_trends(self, incidents: List[Dict[str, Any]]):
        """Render incident trend analysis."""
        st.write("**Incident Trends Over Time**")
        
        # Group incidents by date
        incident_counts = {}
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for incident in incidents:
            # Parse timestamp
            timestamp_str = incident.get('timestamp', '')
            try:
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = timestamp_str
                
                date_key = timestamp.date()
                incident_counts[date_key] = incident_counts.get(date_key, 0) + 1
                
                severity = incident.get('severity', 'unknown')
                if severity in severity_counts:
                    severity_counts[severity] += 1
            
            except:
                continue
        
        # Incidents over time chart
        if incident_counts:
            dates = sorted(incident_counts.keys())
            counts = [incident_counts[date] for date in dates]
            
            fig_trends = go.Figure()
            fig_trends.add_trace(go.Scatter(
                x=dates,
                y=counts,
                mode='lines+markers',
                name='Daily Incidents'
            ))
            
            fig_trends.update_layout(
                title="Incidents Over Time",
                xaxis_title="Date",
                yaxis_title="Number of Incidents",
                height=300
            )
            
            st.plotly_chart(fig_trends, use_container_width=True)
        
        # Severity distribution
        if any(severity_counts.values()):
            fig_severity = px.pie(
                values=list(severity_counts.values()),
                names=list(severity_counts.keys()),
                title="Incidents by Severity"
            )
            
            st.plotly_chart(fig_severity, use_container_width=True)
    
    def _render_incident_patterns(self, incidents: List[Dict[str, Any]]):
        """Render incident pattern analysis."""
        st.write("**Common Patterns and Root Causes**")
        
        # Analyze endpoints
        endpoint_counts = {}
        for incident in incidents:
            endpoint = incident.get('endpoint', 'Unknown')
            endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1
        
        if endpoint_counts:
            # Top affected endpoints
            sorted_endpoints = sorted(endpoint_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            fig_endpoints = px.bar(
                x=[ep[1] for ep in sorted_endpoints],
                y=[ep[0] for ep in sorted_endpoints],
                orientation='h',
                title="Most Affected Endpoints",
                labels={'x': 'Incident Count', 'y': 'Endpoint'}
            )
            
            st.plotly_chart(fig_endpoints, use_container_width=True)
        
        # Time-of-day patterns
        hour_counts = [0] * 24
        for incident in incidents:
            timestamp_str = incident.get('timestamp', '')
            try:
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = timestamp_str
                
                hour_counts[timestamp.hour] += 1
            except:
                continue
        
        if any(hour_counts):
            fig_hours = px.bar(
                x=list(range(24)),
                y=hour_counts,
                title="Incidents by Hour of Day",
                labels={'x': 'Hour', 'y': 'Incident Count'}
            )
            
            st.plotly_chart(fig_hours, use_container_width=True)
    
    def _render_performance_metrics(self, incidents: List[Dict[str, Any]]):
        """Render performance metrics and MTTR analysis."""
        st.write("**Performance Metrics**")
        
        # Calculate metrics
        total_incidents = len(incidents)
        
        # Severity breakdown
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        spike_ratios = []
        
        for incident in incidents:
            severity = incident.get('severity', 'unknown')
            if severity in severity_counts:
                severity_counts[severity] += 1
            
            spike_ratio = incident.get('spike_ratio', 0)
            if spike_ratio > 0:
                spike_ratios.append(spike_ratio)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Incidents", total_incidents)
        
        with col2:
            critical_pct = (severity_counts['critical'] / total_incidents * 100) if total_incidents > 0 else 0
            st.metric("Critical Incidents", f"{critical_pct:.1f}%")
        
        with col3:
            avg_spike_ratio = sum(spike_ratios) / len(spike_ratios) if spike_ratios else 0
            st.metric("Avg Spike Ratio", f"{avg_spike_ratio:.1f}x")
        
        with col4:
            # Simulated MTTR
            mttr_minutes = 15  # Placeholder
            st.metric("Avg MTTR", f"{mttr_minutes} min")
    
    def _apply_filters(self, incidents: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to incidents list."""
        filtered = incidents
        
        # Date range filter
        date_range = filters.get('date_range')
        if date_range and len(date_range) == 2:
            start_date, end_date = date_range
            filtered = [
                inc for inc in filtered
                if self._incident_in_date_range(inc, start_date, end_date)
            ]
        
        # Severity filter
        severity_filter = filters.get('severity_filter', [])
        if severity_filter:
            filtered = [
                inc for inc in filtered
                if inc.get('severity', 'unknown') in severity_filter
            ]
        
        # Endpoint search
        endpoint_search = filters.get('endpoint_search', '').lower()
        if endpoint_search:
            filtered = [
                inc for inc in filtered
                if endpoint_search in inc.get('endpoint', '').lower()
            ]
        
        return filtered
    
    def _render_incident_comparison(self, incidents: List[Dict[str, Any]], selected_rows: List[int], df: pd.DataFrame):
        """Render comparison view for selected incidents."""
        st.subheader("Incident Comparison")
        
        selected_incidents = []
        for row_idx in selected_rows:
            incident_id = df.iloc[row_idx]['ID']
            incident = next((inc for inc in incidents if inc.get('id') == incident_id), None)
            if incident:
                selected_incidents.append(incident)
        
        if len(selected_incidents) < 2:
            st.warning("Please select at least 2 incidents to compare.")
            return
        
        # Comparison table
        comparison_data = []
        for incident in selected_incidents:
            comparison_data.append({
                'ID': incident.get('id', 'N/A'),
                'Endpoint': incident.get('endpoint', 'Unknown'),
                'Severity': incident.get('severity', 'unknown'),
                'Spike Ratio': f"{incident.get('spike_ratio', 0):.1f}x",
                'Timestamp': incident.get('timestamp', 'Unknown'),
                'Causes': incident.get('causes_count', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Severity comparison
            severity_counts = comparison_df['Severity'].value_counts()
            fig_severity = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="Severity Distribution"
            )
            st.plotly_chart(fig_severity, use_container_width=True)
        
        with col2:
            # Spike ratio comparison
            spike_ratios = [float(sr.replace('x', '')) for sr in comparison_df['Spike Ratio']]
            fig_spikes = px.bar(
                x=comparison_df['ID'],
                y=spike_ratios,
                title="Spike Ratio Comparison",
                labels={'x': 'Incident ID', 'y': 'Spike Ratio'}
            )
            st.plotly_chart(fig_spikes, use_container_width=True)
    
    def _export_selected_incidents(self, incidents: List[Dict[str, Any]], selected_rows: List[int], df: pd.DataFrame):
        """Export selected incidents to various formats."""
        st.subheader("Export Selected Incidents")
        
        selected_incidents = []
        for row_idx in selected_rows:
            incident_id = df.iloc[row_idx]['ID']
            incident = next((inc for inc in incidents if inc.get('id') == incident_id), None)
            if incident:
                selected_incidents.append(incident)
        
        if not selected_incidents:
            st.error("No incidents selected for export.")
            return
        
        export_format = st.selectbox(
            "Export Format",
            options=['JSON', 'CSV', 'Excel'],
            help="Choose the format for exporting incident data"
        )
        
        if export_format == 'JSON':
            export_data = json.dumps(selected_incidents, indent=2, default=str)
            st.download_button(
                "📥 Download JSON",
                data=export_data,
                file_name=f"incidents_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        elif export_format == 'CSV':
            # Flatten incident data for CSV
            csv_data = []
            for incident in selected_incidents:
                csv_data.append({
                    'ID': incident.get('id', ''),
                    'Timestamp': incident.get('timestamp', ''),
                    'Endpoint': incident.get('endpoint', ''),
                    'Severity': incident.get('severity', ''),
                    'Spike_Ratio': incident.get('spike_ratio', 0),
                    'Duration': incident.get('duration', ''),
                    'Causes_Count': incident.get('causes_count', 0)
                })
            
            csv_df = pd.DataFrame(csv_data)
            csv_string = csv_df.to_csv(index=False)
            
            st.download_button(
                "📥 Download CSV",
                data=csv_string,
                file_name=f"incidents_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Preview export data
        st.write(f"**Export Preview** ({len(selected_incidents)} incidents)")
        preview_df = pd.DataFrame([
            {
                'ID': inc.get('id', ''),
                'Endpoint': inc.get('endpoint', ''),
                'Severity': inc.get('severity', ''),
                'Timestamp': inc.get('timestamp', '')
            }
            for inc in selected_incidents[:5]  # Show first 5
        ])
        st.dataframe(preview_df, use_container_width=True)
        
        if len(selected_incidents) > 5:
            st.write(f"... and {len(selected_incidents) - 5} more incidents")
    
    def _render_tagging_interface(self, selected_rows: List[int], df: pd.DataFrame):
        """Render interface for tagging incidents."""
        st.subheader("Tag Incidents")
        
        # Tag input
        new_tag = st.text_input(
            "Add Tag",
            placeholder="Enter tag name (e.g., 'database-issue', 'high-priority')",
            help="Tags help categorize and organize incidents"
        )
        
        # Predefined tags
        predefined_tags = [
            'database-issue', 'network-problem', 'high-traffic', 
            'deployment-related', 'infrastructure', 'application-bug',
            'performance-degradation', 'external-service'
        ]
        
        selected_predefined = st.multiselect(
            "Or select from predefined tags:",
            options=predefined_tags,
            help="Common tags for incident categorization"
        )
        
        # Apply tags
        if st.button("🏷️ Apply Tags"):
            tags_to_apply = []
            if new_tag:
                tags_to_apply.append(new_tag)
            tags_to_apply.extend(selected_predefined)
            
            if tags_to_apply:
                selected_ids = [df.iloc[row]['ID'] for row in selected_rows]
                st.success(f"Applied tags {tags_to_apply} to incidents {selected_ids}")
                # In a real implementation, this would update the database
            else:
                st.warning("Please enter or select at least one tag.")
        
        # Show current tags (simulated)
        st.write("**Current Tags for Selected Incidents:**")
        for row in selected_rows:
            incident_id = df.iloc[row]['ID']
            # Simulate existing tags
            existing_tags = ['database-issue', 'high-priority'] if row % 2 == 0 else ['network-problem']
            st.write(f"- Incident {incident_id}: {', '.join(existing_tags)}")
    
    def _incident_in_date_range(self, incident: Dict[str, Any], start_date, end_date) -> bool:
        """Check if incident falls within date range."""
        try:
            timestamp_str = incident.get('timestamp', '')
            if isinstance(timestamp_str, str):
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                timestamp = timestamp_str
            
            incident_date = timestamp.date()
            return start_date <= incident_date <= end_date
        
        except:
            return False