"""
Root Cause Analysis (RCA) Analyzer for the Latency Spike Investigator.

This module provides intelligent analysis of correlation results to identify
potential root causes and generate actionable recommendations.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
from enum import Enum

from models import CorrelationResult, RootCause, SpikeEvent
from clients.base import KnowledgeBaseClient, HistoricalIncident


class AnalysisConfidence(Enum):
    """Confidence levels for RCA analysis."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class RecommendationTemplate:
    """Template for generating recommendations based on patterns."""
    category: str
    pattern_indicators: List[str]
    description_template: str
    actions: List[str]
    confidence_boost: float = 0.0
    priority: int = 1


@dataclass
class AnalysisResult:
    """Complete RCA analysis result with causes and recommendations."""
    correlation_result: CorrelationResult
    identified_causes: List[RootCause]
    confidence_level: AnalysisConfidence
    analysis_duration_ms: float
    knowledge_base_matches: int
    fallback_used: bool = False


class RCAAnalyzer:
    """
    Root Cause Analysis engine that correlates spike events with historical patterns
    and generates actionable recommendations.
    """
    
    def __init__(self, knowledge_base_client: KnowledgeBaseClient):
        """
        Initialize the RCA analyzer.
        
        Args:
            knowledge_base_client: Client for accessing historical incident data
        """
        self.kb_client = knowledge_base_client
        self.logger = logging.getLogger(__name__)
        
        # Predefined recommendation templates for common patterns
        self._recommendation_templates = self._initialize_templates()
        
        # Correlation thresholds for different confidence levels
        self._confidence_thresholds = {
            AnalysisConfidence.VERY_HIGH: 0.85,
            AnalysisConfidence.HIGH: 0.65,
            AnalysisConfidence.MEDIUM: 0.45,
            AnalysisConfidence.LOW: 0.0
        }
    
    def _initialize_templates(self) -> List[RecommendationTemplate]:
        """Initialize predefined recommendation templates."""
        return [
            RecommendationTemplate(
                category="database",
                pattern_indicators=["connection_pool_utilization", "avg_query_time", "lock_wait_time"],
                description_template="Database performance degradation detected with {indicators}",
                actions=[
                    "Check database connection pool utilization",
                    "Review slow query logs for the incident timeframe",
                    "Monitor database lock contention",
                    "Consider scaling database resources",
                    "Review recent database schema changes"
                ],
                confidence_boost=0.1,
                priority=1
            ),
            RecommendationTemplate(
                category="network",
                pattern_indicators=["network_latency", "packet_loss", "bandwidth_utilization"],
                description_template="Network performance issues detected with {indicators}",
                actions=[
                    "Check network latency to upstream services",
                    "Monitor packet loss rates",
                    "Review bandwidth utilization patterns",
                    "Verify DNS resolution times",
                    "Check for DDoS or unusual traffic patterns"
                ],
                confidence_boost=0.1,
                priority=2
            ),
            RecommendationTemplate(
                category="application",
                pattern_indicators=["error_rate", "requests_per_minute", "response_time"],
                description_template="Application performance issues detected with {indicators}",
                actions=[
                    "Review application error logs",
                    "Check memory and CPU utilization",
                    "Monitor garbage collection metrics",
                    "Review recent code deployments",
                    "Check for resource leaks"
                ],
                confidence_boost=0.05,
                priority=3
            ),
            RecommendationTemplate(
                category="infrastructure",
                pattern_indicators=["cpu_utilization", "memory_usage", "disk_io"],
                description_template="Infrastructure resource constraints detected with {indicators}",
                actions=[
                    "Monitor CPU and memory utilization",
                    "Check disk I/O performance",
                    "Review system resource limits",
                    "Consider horizontal or vertical scaling",
                    "Check for resource contention"
                ],
                confidence_boost=0.05,
                priority=4
            ),
            RecommendationTemplate(
                category="external",
                pattern_indicators=["third_party_latency", "external_error_rate"],
                description_template="External service dependency issues detected with {indicators}",
                actions=[
                    "Check third-party service status pages",
                    "Monitor external API response times",
                    "Review rate limiting and circuit breaker configurations",
                    "Implement fallback mechanisms",
                    "Contact external service providers if needed"
                ],
                confidence_boost=0.1,
                priority=2
            )
        ]
    
    async def analyze_incident(self, correlation_result: CorrelationResult) -> AnalysisResult:
        """
        Perform comprehensive root cause analysis on a correlation result.
        
        Args:
            correlation_result: The correlation analysis result to analyze
            
        Returns:
            Complete analysis result with identified causes and recommendations
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Query knowledge base for similar incidents
            similar_incidents = await self._query_knowledge_base(correlation_result)
            
            # Step 2: Analyze correlation patterns
            pattern_causes = self._analyze_correlation_patterns(correlation_result)
            
            # Step 3: Generate causes from historical incidents
            historical_causes = self._generate_historical_causes(similar_incidents, correlation_result)
            
            # Step 4: Combine and rank all potential causes
            all_causes = pattern_causes + historical_causes
            ranked_causes = self._rank_causes(all_causes, correlation_result)
            
            # Step 5: Generate fallback recommendations if needed
            if not ranked_causes or max(cause.confidence_score for cause in ranked_causes) < 0.3:
                fallback_causes = self._generate_fallback_recommendations(correlation_result)
                ranked_causes.extend(fallback_causes)
                fallback_used = True
            else:
                fallback_used = False
            
            # Step 6: Determine overall confidence level
            confidence_level = self._calculate_confidence_level(ranked_causes, len(similar_incidents))
            
            analysis_duration = (datetime.now() - start_time).total_seconds() * 1000
            
            return AnalysisResult(
                correlation_result=correlation_result,
                identified_causes=ranked_causes[:5],  # Top 5 causes
                confidence_level=confidence_level,
                analysis_duration_ms=analysis_duration,
                knowledge_base_matches=len(similar_incidents),
                fallback_used=fallback_used
            )
            
        except Exception as e:
            self.logger.error(f"Error during RCA analysis: {str(e)}")
            # Return fallback analysis in case of error
            fallback_causes = self._generate_fallback_recommendations(correlation_result)
            analysis_duration = (datetime.now() - start_time).total_seconds() * 1000
            
            return AnalysisResult(
                correlation_result=correlation_result,
                identified_causes=fallback_causes,
                confidence_level=AnalysisConfidence.LOW,
                analysis_duration_ms=analysis_duration,
                knowledge_base_matches=0,
                fallback_used=True
            )
    
    async def _query_knowledge_base(self, correlation_result: CorrelationResult) -> List[HistoricalIncident]:
        """Query the knowledge base for similar historical incidents."""
        try:
            if not self.kb_client.is_connected:
                await self.kb_client.connect()
            
            similar_incidents = await self.kb_client.search_similar_incidents(
                correlation_result, 
                similarity_threshold=0.6
            )
            
            self.logger.info(f"Found {len(similar_incidents)} similar incidents in knowledge base")
            return similar_incidents
            
        except Exception as e:
            self.logger.warning(f"Failed to query knowledge base: {str(e)}")
            return []
    
    def _analyze_correlation_patterns(self, correlation_result: CorrelationResult) -> List[RootCause]:
        """Analyze correlation patterns to identify potential causes."""
        causes = []
        strong_correlations = correlation_result.get_strongest_correlations(threshold=0.5)
        
        for template in self._recommendation_templates:
            matching_indicators = []
            total_correlation_strength = 0.0
            
            # Check which indicators from this template are present in correlations
            for indicator in template.pattern_indicators:
                for metric_name, correlation_score in strong_correlations.items():
                    if indicator.lower() in metric_name.lower():
                        matching_indicators.append(f"{metric_name} (correlation: {correlation_score:.2f})")
                        total_correlation_strength += abs(correlation_score)
            
            # If we have matching indicators, create a root cause
            if matching_indicators:
                confidence = min(0.9, (total_correlation_strength / len(matching_indicators)) + template.confidence_boost)
                
                description = template.description_template.format(
                    indicators=", ".join(matching_indicators)
                )
                
                cause = RootCause(
                    category=template.category,
                    description=description,
                    confidence_score=confidence,
                    supporting_evidence=matching_indicators,
                    recommended_actions=template.actions.copy(),
                    priority=template.priority
                )
                
                causes.append(cause)
        
        return causes
    
    def _generate_historical_causes(
        self, 
        similar_incidents: List[HistoricalIncident], 
        correlation_result: CorrelationResult
    ) -> List[RootCause]:
        """Generate root causes based on similar historical incidents."""
        causes = []
        
        for incident in similar_incidents:
            # Calculate confidence based on similarity and historical success
            base_confidence = 0.7  # Base confidence for historical matches
            
            # Adjust confidence based on resolution time (faster resolution = higher confidence)
            time_factor = max(0.1, 1.0 - (incident.resolution_time_minutes / 300.0))  # 5 hours max
            confidence = min(0.95, base_confidence * time_factor)
            
            # Determine category from incident tags
            category = "application"  # default
            if any(tag in ["database", "db", "sql"] for tag in incident.tags):
                category = "database"
            elif any(tag in ["network", "latency", "ddos"] for tag in incident.tags):
                category = "network"
            elif any(tag in ["infrastructure", "cpu", "memory"] for tag in incident.tags):
                category = "infrastructure"
            elif any(tag in ["external", "third_party", "api"] for tag in incident.tags):
                category = "external"
            
            # Create supporting evidence from incident data
            evidence = [
                f"Similar incident: {incident.incident_id}",
                f"Historical resolution time: {incident.resolution_time_minutes} minutes",
                f"Previous root cause: {incident.root_cause}"
            ]
            
            # Add correlation evidence
            for metric, score in correlation_result.correlation_scores.items():
                if abs(score) > 0.4:
                    evidence.append(f"Current correlation: {metric} ({score:.2f})")
            
            # Generate recommended actions based on historical resolution
            actions = [
                f"Review historical incident {incident.incident_id}",
                f"Consider similar resolution: {incident.resolution}",
                "Monitor the same metrics that were affected previously"
            ]
            
            # Add specific actions based on category
            if category == "database":
                actions.extend([
                    "Check database connection pools and query performance",
                    "Review database indexes and query optimization"
                ])
            elif category == "network":
                actions.extend([
                    "Monitor network latency and packet loss",
                    "Check for DDoS protection and traffic routing"
                ])
            
            cause = RootCause(
                category=category,
                description=f"Similar to historical incident: {incident.description}",
                confidence_score=confidence,
                supporting_evidence=evidence,
                recommended_actions=actions,
                priority=1  # Historical matches get high priority
            )
            
            causes.append(cause)
        
        return causes
    
    def _rank_causes(self, causes: List[RootCause], correlation_result: CorrelationResult) -> List[RootCause]:
        """Rank and deduplicate root causes by confidence and relevance."""
        if not causes:
            return []
        
        # Remove duplicates based on category and similar descriptions
        unique_causes = []
        seen_categories = set()
        
        # Sort by confidence score first
        sorted_causes = sorted(causes, key=lambda x: x.confidence_score, reverse=True)
        
        for cause in sorted_causes:
            # Allow multiple causes per category if they're significantly different
            category_key = f"{cause.category}_{cause.description[:50]}"
            
            if category_key not in seen_categories:
                unique_causes.append(cause)
                seen_categories.add(category_key)
        
        # Final ranking considers confidence, priority, and spike severity
        spike_severity_multiplier = {
            "critical": 1.2,
            "high": 1.1,
            "medium": 1.0,
            "low": 0.9
        }.get(correlation_result.spike_event.severity.value, 1.0)
        
        for cause in unique_causes:
            # Adjust confidence based on spike severity
            cause.confidence_score = min(0.99, cause.confidence_score * spike_severity_multiplier)
        
        return sorted(unique_causes, key=lambda x: (x.confidence_score, -x.priority), reverse=True)
    
    def _generate_fallback_recommendations(self, correlation_result: CorrelationResult) -> List[RootCause]:
        """Generate generic fallback recommendations when no specific patterns are identified."""
        spike_event = correlation_result.spike_event
        
        # Generic recommendations based on spike characteristics
        fallback_causes = []
        
        # High latency spike - generic investigation steps
        if spike_event.spike_ratio > 3.0:  # Spike is 3x baseline
            cause = RootCause(
                category="application",
                description=f"Significant latency spike detected on {spike_event.endpoint} "
                           f"({spike_event.spike_latency:.0f}ms vs {spike_event.baseline_latency:.0f}ms baseline)",
                confidence_score=0.4,
                supporting_evidence=[
                    f"Spike ratio: {spike_event.spike_ratio:.1f}x baseline",
                    f"Duration: {spike_event.duration.total_seconds():.0f} seconds",
                    f"Severity: {spike_event.severity.value}"
                ],
                recommended_actions=[
                    "Check application logs for errors during the incident timeframe",
                    "Monitor system resource utilization (CPU, memory, disk)",
                    "Review recent deployments or configuration changes",
                    "Check database query performance and connection pools",
                    "Verify network connectivity to external dependencies",
                    "Review load balancer and caching layer performance"
                ],
                priority=3
            )
            fallback_causes.append(cause)
        
        # If we have some correlation data, provide more specific guidance
        if correlation_result.correlation_scores:
            strongest_correlation = max(
                correlation_result.correlation_scores.items(),
                key=lambda x: abs(x[1])
            )
            
            metric_name, correlation_score = strongest_correlation
            
            if abs(correlation_score) > 0.3:
                cause = RootCause(
                    category="infrastructure",
                    description=f"Potential correlation detected with {metric_name}",
                    confidence_score=0.3 + abs(correlation_score) * 0.2,
                    supporting_evidence=[
                        f"Strongest correlation: {metric_name} ({correlation_score:.2f})",
                        f"Analysis confidence: {correlation_result.confidence:.2f}"
                    ],
                    recommended_actions=[
                        f"Investigate {metric_name} patterns during the incident",
                        "Compare current metrics with historical baselines",
                        "Check for any anomalies in related system components",
                        "Review monitoring alerts from the same timeframe"
                    ],
                    priority=2
                )
                fallback_causes.append(cause)
        
        # Always provide a general investigation checklist
        general_cause = RootCause(
            category="application",
            description="General latency spike investigation checklist",
            confidence_score=0.2,
            supporting_evidence=[
                "No specific patterns identified",
                "Providing general troubleshooting guidance"
            ],
            recommended_actions=[
                "Review application and system logs",
                "Check recent deployments and configuration changes",
                "Monitor resource utilization trends",
                "Verify external service dependencies",
                "Review caching and database performance",
                "Check for any scheduled maintenance or batch jobs"
            ],
            priority=5
        )
        fallback_causes.append(general_cause)
        
        return fallback_causes
    
    def _calculate_confidence_level(self, causes: List[RootCause], kb_matches: int) -> AnalysisConfidence:
        """Calculate overall confidence level for the analysis."""
        if not causes:
            return AnalysisConfidence.LOW
        
        max_confidence = max(cause.confidence_score for cause in causes)
        avg_confidence = sum(cause.confidence_score for cause in causes) / len(causes)
        
        # Boost confidence if we have knowledge base matches
        kb_boost = min(0.2, kb_matches * 0.05)
        adjusted_confidence = max_confidence + kb_boost
        
        if adjusted_confidence >= self._confidence_thresholds[AnalysisConfidence.VERY_HIGH]:
            return AnalysisConfidence.VERY_HIGH
        elif adjusted_confidence >= self._confidence_thresholds[AnalysisConfidence.HIGH]:
            return AnalysisConfidence.HIGH
        elif adjusted_confidence >= self._confidence_thresholds[AnalysisConfidence.MEDIUM]:
            return AnalysisConfidence.MEDIUM
        else:
            return AnalysisConfidence.LOW
    
    async def get_recommendation_feedback(self, cause_id: str, effectiveness: float) -> bool:
        """
        Record feedback on recommendation effectiveness for future improvements.
        
        Args:
            cause_id: Identifier for the root cause recommendation
            effectiveness: Effectiveness score (0.0 to 1.0)
            
        Returns:
            True if feedback was recorded successfully
        """
        try:
            # In a real implementation, this would update ML models or recommendation weights
            self.logger.info(f"Recorded feedback for cause {cause_id}: effectiveness {effectiveness}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to record feedback: {str(e)}")
            return False