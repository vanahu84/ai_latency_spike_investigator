"""
Main Controller for the Latency Spike Root Cause Investigator.

This module provides the central orchestration logic that coordinates all components
from spike detection through recommendation generation, with comprehensive error
handling and graceful degradation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from models import SpikeEvent, CorrelationResult, RootCause, Metric, SeverityLevel
from services.spike_detector import SpikeDetector, ThresholdConfig, SystemStatus
from services.metric_correlator import MetricCorrelator, CorrelationConfig
from services.rca_analyzer import RCAAnalyzer, AnalysisResult
from services.gemini_ai_engine import GeminiAIEngine, EnhancedRecommendation
from clients.manager import MCPClientManager, create_default_mock_manager
from storage.data_access import DataAccessLayer
from config import config


logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Status of the complete workflow execution."""
    PENDING = "pending"
    DETECTING = "detecting"
    CORRELATING = "correlating"
    ANALYZING = "analyzing"
    ENHANCING = "enhancing"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkflowResult:
    """Complete result of the spike investigation workflow."""
    
    workflow_id: str
    status: WorkflowStatus
    spike_event: Optional[SpikeEvent] = None
    correlation_result: Optional[CorrelationResult] = None
    analysis_result: Optional[AnalysisResult] = None
    enhanced_recommendations: List[EnhancedRecommendation] = field(default_factory=list)
    storage_ids: Dict[str, int] = field(default_factory=dict)
    
    # Timing and error information
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    error_message: Optional[str] = None
    
    # Component status tracking
    component_status: Dict[str, str] = field(default_factory=dict)
    fallback_used: bool = False


class LatencySpikeController:
    """
    Main controller that orchestrates the complete workflow from spike detection
    to recommendation generation with error handling and graceful degradation.
    """
    
    def __init__(self, 
                 client_manager: Optional[MCPClientManager] = None,
                 data_access: Optional[DataAccessLayer] = None,
                 spike_detector: Optional[SpikeDetector] = None,
                 threshold_config: Optional[ThresholdConfig] = None,
                 correlation_config: Optional[CorrelationConfig] = None):
        """
        Initialize the controller with optional custom components.
        
        Args:
            client_manager: MCP client manager for data sources
            data_access: Data access layer for storage
            spike_detector: Spike detection service
            threshold_config: Configuration for spike detection thresholds
            correlation_config: Configuration for correlation analysis
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize components
        self.client_manager = client_manager or create_default_mock_manager()
        self.data_access = data_access or DataAccessLayer()
        
        # Initialize services
        self.spike_detector = spike_detector or SpikeDetector(threshold_config)
        self.correlation_config = correlation_config or CorrelationConfig()
        
        # These will be initialized after client manager is ready
        self.metric_correlator: Optional[MetricCorrelator] = None
        self.rca_analyzer: Optional[RCAAnalyzer] = None
        self.ai_engine: Optional[GeminiAIEngine] = None
        
        # State tracking
        self._initialized = False
        self._running_workflows: Dict[str, WorkflowResult] = {}
        self._workflow_counter = 0
        
        # Health monitoring
        self._last_health_check: Optional[datetime] = None
        self._component_health: Dict[str, bool] = {}
    
    async def initialize(self) -> bool:
        """
        Initialize all components and establish connections.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return True
        
        self.logger.info("Initializing Latency Spike Controller...")
        
        try:
            # Initialize data access layer
            self.data_access.initialize()
            self.logger.info("Data access layer initialized")
            
            # Initialize MCP client manager
            client_init_success = await self.client_manager.initialize()
            if not client_init_success:
                self.logger.warning("MCP client manager initialization failed - using fallback mode")
            
            # Start health monitoring for clients
            await self.client_manager.start_health_monitoring()
            
            # Initialize services that depend on clients
            self.metric_correlator = MetricCorrelator(
                self.client_manager, 
                self.correlation_config
            )
            
            kb_client = self.client_manager.get_knowledge_base_client()
            if kb_client:
                self.rca_analyzer = RCAAnalyzer(kb_client)
                self.logger.info("RCA analyzer initialized with knowledge base")
            else:
                self.logger.warning("No knowledge base client available - RCA will use fallback mode")
                # Create a mock knowledge base client for fallback
                from clients.mock import MockKnowledgeBaseClient
                mock_kb = MockKnowledgeBaseClient("fallback_kb", {})
                await mock_kb.connect()
                self.rca_analyzer = RCAAnalyzer(mock_kb)
            
            # Initialize AI engine
            self.ai_engine = GeminiAIEngine()
            if self.ai_engine.is_available():
                self.logger.info("Gemini AI engine initialized")
            else:
                self.logger.warning("Gemini AI engine not available - using fallback recommendations")
            
            self._initialized = True
            self.logger.info("Controller initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Controller initialization failed: {e}")
            return False
    
    async def process_metrics(self, metrics: List[Metric]) -> List[WorkflowResult]:
        """
        Process incoming metrics and execute complete workflow for any detected spikes.
        
        Args:
            metrics: List of metric data points to analyze
            
        Returns:
            List of workflow results for any spikes detected and processed
        """
        if not self._initialized:
            await self.initialize()
        
        if not metrics:
            return []
        
        self.logger.debug(f"Processing {len(metrics)} metrics")
        
        try:
            # Step 1: Detect spikes
            detected_spikes = self.spike_detector.detect_spikes(metrics)
            
            if not detected_spikes:
                self.logger.debug("No spikes detected in metrics")
                return []
            
            self.logger.info(f"Detected {len(detected_spikes)} spikes")
            
            # Step 2: Process each spike through the complete workflow
            workflow_results = []
            for spike in detected_spikes:
                try:
                    result = await self.execute_spike_workflow(spike)
                    workflow_results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to process spike workflow: {e}")
                    # Create a failed workflow result
                    failed_result = WorkflowResult(
                        workflow_id=self._generate_workflow_id(),
                        status=WorkflowStatus.FAILED,
                        spike_event=spike,
                        error_message=str(e)
                    )
                    workflow_results.append(failed_result)
            
            return workflow_results
            
        except Exception as e:
            self.logger.error(f"Error processing metrics: {e}")
            return []
    
    async def execute_spike_workflow(self, spike_event: SpikeEvent) -> WorkflowResult:
        """
        Execute the complete workflow for a single spike event.
        
        Args:
            spike_event: The detected spike to analyze
            
        Returns:
            Complete workflow result
        """
        workflow_id = self._generate_workflow_id()
        result = WorkflowResult(
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            spike_event=spike_event
        )
        
        self._running_workflows[workflow_id] = result
        
        try:
            self.logger.info(f"Starting workflow {workflow_id} for spike on {spike_event.endpoint}")
            
            # Step 1: Correlation Analysis
            result.status = WorkflowStatus.CORRELATING
            correlation_result = await self._execute_correlation_analysis(spike_event, result)
            result.correlation_result = correlation_result
            
            # Step 2: Root Cause Analysis
            result.status = WorkflowStatus.ANALYZING
            analysis_result = await self._execute_rca_analysis(correlation_result, result)
            result.analysis_result = analysis_result
            
            # Step 3: AI Enhancement (if available)
            result.status = WorkflowStatus.ENHANCING
            enhanced_recommendations = await self._execute_ai_enhancement(
                analysis_result.identified_causes, result
            )
            result.enhanced_recommendations = enhanced_recommendations
            
            # Step 4: Store Results
            result.status = WorkflowStatus.STORING
            storage_ids = await self._store_workflow_results(result)
            result.storage_ids = storage_ids
            
            # Complete workflow
            result.status = WorkflowStatus.COMPLETED
            result.end_time = datetime.now()
            result.duration_ms = (result.end_time - result.start_time).total_seconds() * 1000
            
            self.logger.info(
                f"Workflow {workflow_id} completed successfully in {result.duration_ms:.0f}ms"
            )
            
        except Exception as e:
            result.status = WorkflowStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.duration_ms = (result.end_time - result.start_time).total_seconds() * 1000
            
            self.logger.error(f"Workflow {workflow_id} failed: {e}")
        
        finally:
            # Clean up from running workflows
            if workflow_id in self._running_workflows:
                del self._running_workflows[workflow_id]
        
        return result
    
    async def _execute_correlation_analysis(
        self, spike_event: SpikeEvent, workflow_result: WorkflowResult
    ) -> CorrelationResult:
        """Execute correlation analysis with error handling."""
        try:
            if not self.metric_correlator:
                raise RuntimeError("Metric correlator not initialized")
            
            correlation_result = await self.metric_correlator.correlate_metrics(spike_event)
            workflow_result.component_status['correlation'] = 'success'
            
            self.logger.debug(
                f"Correlation analysis completed with confidence {correlation_result.confidence:.2f}"
            )
            
            return correlation_result
            
        except Exception as e:
            workflow_result.component_status['correlation'] = f'failed: {str(e)}'
            workflow_result.fallback_used = True
            
            self.logger.warning(f"Correlation analysis failed, using fallback: {e}")
            
            # Create fallback correlation result
            return CorrelationResult(
                spike_event=spike_event,
                network_metrics={},
                db_metrics={},
                correlation_scores={},
                confidence=0.1,
                analysis_timestamp=datetime.now()
            )
    
    async def _execute_rca_analysis(
        self, correlation_result: CorrelationResult, workflow_result: WorkflowResult
    ) -> AnalysisResult:
        """Execute RCA analysis with error handling."""
        try:
            if not self.rca_analyzer:
                raise RuntimeError("RCA analyzer not initialized")
            
            analysis_result = await self.rca_analyzer.analyze_incident(correlation_result)
            workflow_result.component_status['rca'] = 'success'
            
            self.logger.debug(
                f"RCA analysis completed with {len(analysis_result.identified_causes)} causes"
            )
            
            return analysis_result
            
        except Exception as e:
            workflow_result.component_status['rca'] = f'failed: {str(e)}'
            workflow_result.fallback_used = True
            
            self.logger.warning(f"RCA analysis failed, using fallback: {e}")
            
            # Create fallback analysis result
            fallback_cause = RootCause(
                category="application",
                description=f"Latency spike detected on {correlation_result.spike_event.endpoint} "
                           f"(analysis failed: {str(e)})",
                confidence_score=0.2,
                supporting_evidence=[
                    f"Spike ratio: {correlation_result.spike_event.spike_ratio:.1f}x",
                    "Automated analysis unavailable"
                ],
                recommended_actions=[
                    "Check application logs for errors",
                    "Monitor system resource utilization",
                    "Review recent deployments"
                ],
                priority=3
            )
            
            from services.rca_analyzer import AnalysisResult, AnalysisConfidence
            return AnalysisResult(
                correlation_result=correlation_result,
                identified_causes=[fallback_cause],
                confidence_level=AnalysisConfidence.LOW,
                analysis_duration_ms=0.0,
                knowledge_base_matches=0,
                fallback_used=True
            )
    
    async def _execute_ai_enhancement(
        self, basic_causes: List[RootCause], workflow_result: WorkflowResult
    ) -> List[EnhancedRecommendation]:
        """Execute AI enhancement with error handling."""
        try:
            if not self.ai_engine or not self.ai_engine.is_available():
                workflow_result.component_status['ai_enhancement'] = 'unavailable'
                workflow_result.fallback_used = True
                
                # Create fallback enhanced recommendations
                enhanced_recs = []
                for cause in basic_causes:
                    enhanced_rec = self.ai_engine._create_fallback_enhancement(cause) if self.ai_engine else None
                    if enhanced_rec:
                        enhanced_recs.append(enhanced_rec)
                
                return enhanced_recs
            
            enhanced_recommendations = await self.ai_engine.enhance_recommendations(basic_causes)
            workflow_result.component_status['ai_enhancement'] = 'success'
            
            self.logger.debug(f"AI enhancement completed for {len(enhanced_recommendations)} recommendations")
            
            return enhanced_recommendations
            
        except Exception as e:
            workflow_result.component_status['ai_enhancement'] = f'failed: {str(e)}'
            workflow_result.fallback_used = True
            
            self.logger.warning(f"AI enhancement failed, using fallback: {e}")
            
            # Create fallback enhanced recommendations
            enhanced_recs = []
            for cause in basic_causes:
                if self.ai_engine:
                    enhanced_rec = self.ai_engine._create_fallback_enhancement(cause)
                    enhanced_recs.append(enhanced_rec)
            
            return enhanced_recs
    
    async def _store_workflow_results(self, workflow_result: WorkflowResult) -> Dict[str, int]:
        """Store workflow results with error handling."""
        try:
            # Extract recommendations from enhanced recommendations
            recommendations = []
            if workflow_result.enhanced_recommendations:
                recommendations = [er.original_cause for er in workflow_result.enhanced_recommendations]
            elif workflow_result.analysis_result:
                recommendations = workflow_result.analysis_result.identified_causes
            
            storage_ids = self.data_access.store_incident(
                workflow_result.spike_event,
                workflow_result.correlation_result,
                recommendations
            )
            
            workflow_result.component_status['storage'] = 'success'
            
            self.logger.debug(f"Workflow results stored with IDs: {storage_ids}")
            
            return storage_ids
            
        except Exception as e:
            workflow_result.component_status['storage'] = f'failed: {str(e)}'
            
            self.logger.error(f"Failed to store workflow results: {e}")
            return {}
    
    def _generate_workflow_id(self) -> str:
        """Generate a unique workflow ID."""
        self._workflow_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"workflow_{timestamp}_{self._workflow_counter:04d}"
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        health_status = {
            'controller': {
                'initialized': self._initialized,
                'running_workflows': len(self._running_workflows),
                'total_workflows_processed': self._workflow_counter
            },
            'components': {},
            'overall_status': 'healthy',
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Check spike detector health
            detector_status = self.spike_detector.get_current_status()
            health_status['components']['spike_detector'] = {
                'healthy': detector_status.is_healthy,
                'active_spikes': detector_status.active_spikes,
                'last_check': detector_status.last_check_time.isoformat() if detector_status.last_check_time else None,
                'error': detector_status.error_message
            }
            
            # Check client manager health
            if self.client_manager:
                client_health = await self.client_manager.health_check_all()
                health_status['components']['clients'] = {
                    client_key: {
                        'status': result.status.value,
                        'response_time_ms': result.response_time_ms,
                        'error': result.error_message
                    }
                    for client_key, result in client_health.items()
                }
            
            # Check data access health
            if self.data_access:
                storage_health = self.data_access.get_system_health()
                health_status['components']['storage'] = storage_health
            
            # Check AI engine health
            if self.ai_engine:
                ai_health = await self.ai_engine.health_check()
                health_status['components']['ai_engine'] = ai_health
            
            # Determine overall status
            component_issues = []
            for component, status in health_status['components'].items():
                if isinstance(status, dict):
                    if not status.get('healthy', True) and status.get('status') != 'healthy':
                        component_issues.append(component)
            
            if component_issues:
                if len(component_issues) > len(health_status['components']) / 2:
                    health_status['overall_status'] = 'unhealthy'
                else:
                    health_status['overall_status'] = 'degraded'
            
            self._last_health_check = datetime.now()
            
        except Exception as e:
            health_status['overall_status'] = 'error'
            health_status['error'] = str(e)
            self.logger.error(f"Health check failed: {e}")
        
        return health_status
    
    def get_running_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Get information about currently running workflows."""
        return {
            workflow_id: {
                'status': result.status.value,
                'start_time': result.start_time.isoformat(),
                'duration_ms': (datetime.now() - result.start_time).total_seconds() * 1000,
                'spike_endpoint': result.spike_event.endpoint if result.spike_event else None,
                'component_status': result.component_status
            }
            for workflow_id, result in self._running_workflows.items()
        }
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the controller and all components."""
        self.logger.info("Shutting down Latency Spike Controller...")
        
        try:
            # Wait for running workflows to complete (with timeout)
            if self._running_workflows:
                self.logger.info(f"Waiting for {len(self._running_workflows)} workflows to complete...")
                
                # Wait up to 30 seconds for workflows to complete
                timeout = 30
                start_time = datetime.now()
                
                while self._running_workflows and (datetime.now() - start_time).total_seconds() < timeout:
                    await asyncio.sleep(1)
                
                if self._running_workflows:
                    self.logger.warning(f"Timeout waiting for workflows: {list(self._running_workflows.keys())}")
            
            # Shutdown client manager
            if self.client_manager:
                await self.client_manager.shutdown()
            
            self._initialized = False
            self.logger.info("Controller shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    # Configuration and management methods
    
    def update_spike_thresholds(self, thresholds: Dict[str, float]) -> bool:
        """Update spike detection thresholds."""
        try:
            self.spike_detector.configure_thresholds(thresholds)
            self.logger.info(f"Updated spike thresholds: {thresholds}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update spike thresholds: {e}")
            return False
    
    def get_recent_incidents(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent incidents from storage."""
        try:
            return self.data_access.get_recent_incidents(limit=limit)
        except Exception as e:
            self.logger.error(f"Failed to get recent incidents: {e}")
            return []
    
    def get_incident_details(self, incident_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific incident."""
        try:
            return self.data_access.get_incident_details(incident_id)
        except Exception as e:
            self.logger.error(f"Failed to get incident details: {e}")
            return None
    
    def store_recommendation_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """Store feedback about recommendation effectiveness."""
        try:
            # Store feedback in database
            feedback_id = self.data_access.store_recommendation_feedback(feedback_data)
            
            self.logger.info(
                f"Stored recommendation feedback: {feedback_data['recommendation_id']} "
                f"- {feedback_data['effectiveness']}"
            )
            
            return feedback_id is not None
            
        except Exception as e:
            self.logger.error(f"Failed to store recommendation feedback: {e}")
            return False
    
    def update_action_completion(self, completion_data: Dict[str, Any]) -> bool:
        """Update the completion status of a recommendation action."""
        try:
            # Store action completion in database
            completion_id = self.data_access.store_action_completion(completion_data)
            
            self.logger.info(
                f"Updated action completion: {completion_data['action_key']} "
                f"- {'completed' if completion_data['completed'] else 'not completed'}"
            )
            
            return completion_id is not None
            
        except Exception as e:
            self.logger.error(f"Failed to update action completion: {e}")
            return False
    
    def get_recommendation_analytics(self) -> Dict[str, Any]:
        """Get analytics about recommendation effectiveness."""
        try:
            return self.data_access.get_recommendation_analytics()
        except Exception as e:
            self.logger.error(f"Failed to get recommendation analytics: {e}")
            return {}