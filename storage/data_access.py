"""
Data Access Layer (DAL) that provides a unified interface to both database and cache.

This module coordinates between SQLite database and Redis cache to provide
efficient data access with proper error handling and fallback mechanisms.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json

from .database import DatabaseManager, DatabaseError
from .cache import CacheManager, CacheError
from models.core import SpikeEvent, CorrelationResult, RootCause, Metric, SeverityLevel


logger = logging.getLogger(__name__)


class DataAccessError(Exception):
    """Custom exception for data access operations."""
    pass


class DataAccessLayer:
    """
    Unified data access layer that coordinates between database and cache.
    
    Provides high-level operations for storing and retrieving application data
    with automatic caching, fallback mechanisms, and error handling.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None, 
                 cache_manager: Optional[CacheManager] = None):
        """Initialize data access layer with optional custom managers."""
        self.db = db_manager or DatabaseManager()
        self.cache = cache_manager or CacheManager()
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the data access layer."""
        if self._initialized:
            return
        
        try:
            # Initialize database
            self.db.initialize_database()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DataAccessError(f"Database initialization failed: {e}")
        
        # Cache initialization is optional - log warning if unavailable
        if not self.cache.is_available():
            logger.warning("Cache is not available - operating without caching")
        else:
            logger.info("Cache is available and ready")
        
        self._initialized = True
    
    def store_incident(self, spike_event: SpikeEvent, correlation: Optional[CorrelationResult] = None,
                      recommendations: Optional[List[RootCause]] = None) -> Dict[str, int]:
        """
        Store a complete incident with spike event, correlation, and recommendations.
        
        Returns a dictionary with the IDs of stored records.
        """
        if not self._initialized:
            self.initialize()
        
        try:
            # Store spike event
            spike_id = self.db.store_spike_event(spike_event)
            result = {'spike_event_id': spike_id}
            
            # Cache the spike event for quick access
            self.cache.cache_spike_event(spike_event)
            
            # Store correlation if provided
            if correlation:
                correlation_id = self.db.store_correlation_result(correlation, spike_id)
                result['correlation_id'] = correlation_id
                
                # Cache correlation result
                self.cache.cache_correlation_result(correlation)
                
                # Store recommendations if provided
                if recommendations:
                    recommendation_ids = []
                    for recommendation in recommendations:
                        rec_id = self.db.store_recommendation(recommendation, correlation_id)
                        recommendation_ids.append(rec_id)
                    result['recommendation_ids'] = recommendation_ids
            
            logger.info(f"Stored incident with spike_id: {spike_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to store incident: {e}")
            raise DataAccessError(f"Failed to store incident: {e}")
    
    def get_recent_incidents(self, limit: int = 50, include_analysis: bool = True) -> List[Dict[str, Any]]:
        """Get recent incidents with optional analysis data."""
        if not self._initialized:
            self.initialize()
        
        try:
            # Try cache first for recent incidents
            if self.cache.is_available():
                cached_incidents = self.cache.get_recent_spikes(limit=limit)
                if cached_incidents and len(cached_incidents) >= min(limit, 10):
                    logger.debug(f"Retrieved {len(cached_incidents)} incidents from cache")
                    return cached_incidents
            
            # Fallback to database
            incidents = self.db.get_recent_spike_events(limit)
            
            # Enrich with analysis data if requested
            if include_analysis:
                enriched_incidents = []
                for incident in incidents:
                    full_incident = self.db.get_spike_event_with_analysis(incident['id'])
                    if full_incident:
                        enriched_incidents.append(full_incident)
                incidents = enriched_incidents
            
            # Cache the results for future requests
            if self.cache.is_available() and incidents:
                for incident in incidents[:10]:  # Cache top 10
                    try:
                        # Convert database format to spike event for caching
                        spike_event = self._dict_to_spike_event(incident)
                        self.cache.cache_spike_event(spike_event)
                    except Exception as e:
                        logger.warning(f"Failed to cache incident: {e}")
            
            return incidents
            
        except Exception as e:
            logger.error(f"Failed to get recent incidents: {e}")
            raise DataAccessError(f"Failed to get recent incidents: {e}")
    
    def get_incident_details(self, spike_event_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific incident."""
        if not self._initialized:
            self.initialize()
        
        try:
            # Get from database (most complete source)
            incident = self.db.get_spike_event_with_analysis(spike_event_id)
            
            if incident:
                logger.debug(f"Retrieved incident details for ID: {spike_event_id}")
            
            return incident
            
        except Exception as e:
            logger.error(f"Failed to get incident details for ID {spike_event_id}: {e}")
            raise DataAccessError(f"Failed to get incident details: {e}")
    
    def update_recommendation_feedback(self, recommendation_id: int, status: str, 
                                     feedback: Optional[str] = None) -> None:
        """Update recommendation status and user feedback."""
        if not self._initialized:
            self.initialize()
        
        valid_statuses = ['pending', 'completed', 'dismissed']
        if status not in valid_statuses:
            raise DataAccessError(f"Invalid status. Must be one of: {valid_statuses}")
        
        try:
            self.db.update_recommendation_status(recommendation_id, status, feedback)
            logger.info(f"Updated recommendation {recommendation_id} status to {status}")
            
        except Exception as e:
            logger.error(f"Failed to update recommendation feedback: {e}")
            raise DataAccessError(f"Failed to update recommendation feedback: {e}")
    
    def cache_metrics(self, metrics: List[Metric], ttl: Optional[int] = None) -> bool:
        """Cache metrics for quick access during correlation analysis."""
        if not metrics:
            return True
        
        try:
            success = self.cache.set_metrics(metrics, ttl)
            if success:
                logger.debug(f"Cached {len(metrics)} metrics")
            return success
            
        except Exception as e:
            logger.warning(f"Failed to cache metrics: {e}")
            return False
    
    def get_cached_metrics(self, source: str, metric_name: str,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get cached metrics for correlation analysis."""
        try:
            metrics = self.cache.get_metrics(source, metric_name, start_time, end_time)
            if metrics:
                logger.debug(f"Retrieved {len(metrics)} cached metrics for {source}:{metric_name}")
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to get cached metrics: {e}")
            return []
    
    def get_historical_patterns(self, category: Optional[str] = None, 
                               similarity_threshold: float = 0.7,
                               limit: int = 50) -> List[Dict[str, Any]]:
        """Get historical incident patterns for knowledge base queries."""
        if not self._initialized:
            self.initialize()
        
        try:
            patterns = self.db.get_historical_patterns(category, limit)
            
            # Filter by similarity if threshold is specified
            if similarity_threshold > 0 and patterns:
                # This is a simplified similarity check - in a real implementation,
                # you might want more sophisticated pattern matching
                filtered_patterns = []
                for pattern in patterns:
                    # Simple similarity based on category and confidence
                    if pattern.get('confidence_score', 0) >= similarity_threshold:
                        filtered_patterns.append(pattern)
                patterns = filtered_patterns
            
            logger.debug(f"Retrieved {len(patterns)} historical patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to get historical patterns: {e}")
            raise DataAccessError(f"Failed to get historical patterns: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive health status of storage systems."""
        health_status = {
            'database': {
                'available': False,
                'error': None,
                'stats': {}
            },
            'cache': {
                'available': False,
                'error': None,
                'stats': {}
            },
            'overall_health': 'unhealthy',
            'timestamp': datetime.now().isoformat()
        }
        
        # Check database health
        try:
            health_status['database']['available'] = self.db.health_check()
            if health_status['database']['available']:
                health_status['database']['stats'] = self.db.get_database_stats()
        except Exception as e:
            health_status['database']['error'] = str(e)
        
        # Check cache health
        try:
            health_status['cache']['available'] = self.cache.health_check()
            if health_status['cache']['available']:
                health_status['cache']['stats'] = self.cache.get_cache_stats()
        except Exception as e:
            health_status['cache']['error'] = str(e)
        
        # Determine overall health
        if health_status['database']['available']:
            if health_status['cache']['available']:
                health_status['overall_health'] = 'healthy'
            else:
                health_status['overall_health'] = 'degraded'  # DB works, cache doesn't
        
        return health_status
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old data to manage storage space."""
        if not self._initialized:
            self.initialize()
        
        cleanup_stats = {
            'spike_events_deleted': 0,
            'cache_keys_cleared': 0
        }
        
        try:
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean up old spike events (this will cascade to correlations and recommendations)
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    DELETE FROM spike_events 
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))
                cleanup_stats['spike_events_deleted'] = cursor.rowcount
                conn.commit()
            
            # Clean up expired cache keys
            if self.cache.is_available():
                cleanup_stats['cache_keys_cleared'] = self.cache.clear_expired_keys()
            
            logger.info(f"Cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            raise DataAccessError(f"Failed to cleanup old data: {e}")
    
    def _dict_to_spike_event(self, data: Dict[str, Any]) -> SpikeEvent:
        """Convert database dictionary to SpikeEvent object."""
        try:
            return SpikeEvent(
                timestamp=datetime.fromisoformat(data['timestamp']),
                endpoint=data['endpoint'],
                severity=SeverityLevel(data['severity']),
                baseline_latency=data['baseline_latency'],
                spike_latency=data['spike_latency'],
                duration=timedelta(seconds=data['duration_seconds']),
                affected_metrics=json.loads(data.get('affected_metrics', '[]')),
                metadata=json.loads(data.get('metadata', '{}'))
            )
        except Exception as e:
            logger.error(f"Failed to convert dict to SpikeEvent: {e}")
            raise DataAccessError(f"Failed to convert data format: {e}")
    
    def store_incidents_batch(self, incidents: List[Tuple[SpikeEvent, Optional[CorrelationResult], 
                                                        Optional[List[RootCause]]]]) -> List[Dict[str, int]]:
        """Store multiple incidents in a batch for better performance."""
        if not self._initialized:
            self.initialize()
        
        if not incidents:
            return []
        
        results = []
        try:
            for spike_event, correlation, recommendations in incidents:
                result = self.store_incident(spike_event, correlation, recommendations)
                results.append(result)
            
            logger.info(f"Stored {len(incidents)} incidents in batch")
            return results
            
        except Exception as e:
            logger.error(f"Failed to store incidents batch: {e}")
            raise DataAccessError(f"Failed to store incidents batch: {e}")
    
    def get_incidents_by_endpoint(self, endpoint: str, hours: int = 24, limit: int = 50) -> List[Dict[str, Any]]:
        """Get incidents for a specific endpoint within a time window."""
        if not self._initialized:
            self.initialize()
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM spike_events 
                    WHERE endpoint = ? AND timestamp > ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (endpoint, cutoff_time.isoformat(), limit))
                
                incidents = [dict(row) for row in cursor.fetchall()]
            
            logger.debug(f"Retrieved {len(incidents)} incidents for endpoint {endpoint}")
            return incidents
            
        except Exception as e:
            logger.error(f"Failed to get incidents for endpoint {endpoint}: {e}")
            raise DataAccessError(f"Failed to get incidents for endpoint: {e}")
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive summary statistics for recent metrics and incidents."""
        if not self._initialized:
            self.initialize()
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Get recent incidents from database
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_incidents,
                        COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_incidents,
                        COUNT(CASE WHEN severity = 'high' THEN 1 END) as high_incidents,
                        COUNT(CASE WHEN severity = 'medium' THEN 1 END) as medium_incidents,
                        COUNT(CASE WHEN severity = 'low' THEN 1 END) as low_incidents,
                        AVG(spike_latency - baseline_latency) as avg_spike_increase,
                        MAX(spike_latency - baseline_latency) as max_spike_increase,
                        COUNT(DISTINCT endpoint) as affected_endpoints
                    FROM spike_events 
                    WHERE timestamp > ?
                """, (cutoff_time.isoformat(),))
                
                stats = dict(cursor.fetchone())
                
                # Get top affected endpoints
                cursor = conn.execute("""
                    SELECT endpoint, COUNT(*) as incident_count
                    FROM spike_events 
                    WHERE timestamp > ?
                    GROUP BY endpoint
                    ORDER BY incident_count DESC
                    LIMIT 5
                """, (cutoff_time.isoformat(),))
                
                stats['top_affected_endpoints'] = [dict(row) for row in cursor.fetchall()]
            
            # Get cache statistics
            cache_stats = self.cache.get_cache_stats() if self.cache.is_available() else {}
            
            # Get database statistics
            db_stats = self.db.get_database_stats()
            
            return {
                'time_window_hours': hours,
                'incident_stats': stats,
                'cache_stats': cache_stats,
                'database_stats': db_stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            raise DataAccessError(f"Failed to get metrics summary: {e}")
    
    def store_recommendation_feedback(self, feedback_data: Dict[str, Any]) -> Optional[int]:
        """Store feedback about recommendation effectiveness."""
        if not self._initialized:
            self.initialize()
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO recommendation_feedback (
                        recommendation_id, incident_id, effectiveness, implementation,
                        time_to_implement, feedback_text, timestamp, user_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback_data.get('recommendation_id'),
                    feedback_data.get('incident_id'),
                    feedback_data.get('effectiveness'),
                    feedback_data.get('implementation'),
                    feedback_data.get('time_to_implement'),
                    feedback_data.get('feedback_text'),
                    feedback_data.get('timestamp', datetime.now().isoformat()),
                    feedback_data.get('user_id', 'anonymous')
                ))
                
                feedback_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Stored recommendation feedback with ID {feedback_id}")
                return feedback_id
                
        except Exception as e:
            logger.error(f"Failed to store recommendation feedback: {e}")
            return None
    
    def store_action_completion(self, completion_data: Dict[str, Any]) -> Optional[int]:
        """Store action completion status."""
        if not self._initialized:
            self.initialize()
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT OR REPLACE INTO action_completions (
                        incident_id, action_key, action_text, completed,
                        timestamp, user_id, completion_notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    completion_data.get('incident_id'),
                    completion_data.get('action_key'),
                    completion_data.get('action_text'),
                    completion_data.get('completed', False),
                    completion_data.get('timestamp', datetime.now().isoformat()),
                    completion_data.get('user', 'anonymous'),
                    completion_data.get('completion_notes', '')
                ))
                
                completion_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Stored action completion with ID {completion_id}")
                return completion_id
                
        except Exception as e:
            logger.error(f"Failed to store action completion: {e}")
            return None
    
    def get_recommendation_analytics(self) -> Dict[str, Any]:
        """Get analytics about recommendation effectiveness."""
        if not self._initialized:
            self.initialize()
        
        try:
            analytics = {}
            
            with self.db.get_connection() as conn:
                # Overall feedback statistics
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_feedback,
                        COUNT(CASE WHEN effectiveness = 'Very Helpful' THEN 1 END) as very_helpful,
                        COUNT(CASE WHEN effectiveness = 'Somewhat Helpful' THEN 1 END) as somewhat_helpful,
                        COUNT(CASE WHEN effectiveness = 'Not Helpful' THEN 1 END) as not_helpful,
                        COUNT(CASE WHEN effectiveness = 'Made it Worse' THEN 1 END) as made_worse,
                        COUNT(CASE WHEN implementation = 'Successfully Implemented' THEN 1 END) as successful_impl,
                        COUNT(CASE WHEN implementation = 'Partially Implemented' THEN 1 END) as partial_impl,
                        COUNT(CASE WHEN implementation = 'Could Not Implement' THEN 1 END) as failed_impl
                    FROM recommendation_feedback
                """)
                
                analytics['feedback_summary'] = dict(cursor.fetchone())
                
                # Implementation time distribution
                cursor = conn.execute("""
                    SELECT time_to_implement, COUNT(*) as count
                    FROM recommendation_feedback
                    WHERE time_to_implement IS NOT NULL
                    GROUP BY time_to_implement
                    ORDER BY count DESC
                """)
                
                analytics['implementation_time_distribution'] = [dict(row) for row in cursor.fetchall()]
                
                # Action completion rates
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_actions,
                        COUNT(CASE WHEN completed = 1 THEN 1 END) as completed_actions,
                        ROUND(
                            (COUNT(CASE WHEN completed = 1 THEN 1 END) * 100.0 / COUNT(*)), 2
                        ) as completion_rate
                    FROM action_completions
                """)
                
                analytics['action_completion'] = dict(cursor.fetchone())
                
                # Recent feedback trends (last 30 days)
                cutoff_date = datetime.now() - timedelta(days=30)
                cursor = conn.execute("""
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as feedback_count,
                        AVG(CASE 
                            WHEN effectiveness = 'Very Helpful' THEN 4
                            WHEN effectiveness = 'Somewhat Helpful' THEN 3
                            WHEN effectiveness = 'Not Helpful' THEN 2
                            WHEN effectiveness = 'Made it Worse' THEN 1
                            ELSE 0
                        END) as avg_effectiveness_score
                    FROM recommendation_feedback
                    WHERE timestamp > ?
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                    LIMIT 30
                """, (cutoff_date.isoformat(),))
                
                analytics['recent_trends'] = [dict(row) for row in cursor.fetchall()]
            
            analytics['timestamp'] = datetime.now().isoformat()
            logger.debug("Retrieved recommendation analytics")
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get recommendation analytics: {e}")
            return {}
    
    def get_action_completion_status(self, incident_id: str, action_key: str) -> bool:
        """Get the completion status of a specific action."""
        if not self._initialized:
            self.initialize()
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT completed FROM action_completions
                    WHERE incident_id = ? AND action_key = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (incident_id, action_key))
                
                result = cursor.fetchone()
                return bool(result[0]) if result else False
                
        except Exception as e:
            logger.error(f"Failed to get action completion status: {e}")
            return False
    
    def get_feedback_for_incident(self, incident_id: str) -> List[Dict[str, Any]]:
        """Get all feedback for a specific incident."""
        if not self._initialized:
            self.initialize()
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM recommendation_feedback
                    WHERE incident_id = ?
                    ORDER BY timestamp DESC
                """, (incident_id,))
                
                feedback = [dict(row) for row in cursor.fetchall()]
                
                logger.debug(f"Retrieved {len(feedback)} feedback entries for incident {incident_id}")
                return feedback
                
        except Exception as e:
            logger.error(f"Failed to get feedback for incident {incident_id}: {e}")
            return []