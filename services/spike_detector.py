"""
Spike Detection Engine for the Latency Spike Root Cause Investigator.

This module implements the SpikeDetector class that monitors metrics for latency spikes,
aggregates consecutive spikes to prevent alert fatigue, and calculates severity levels.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import statistics
from dataclasses import dataclass, field

from models.core import Metric, SpikeEvent, SeverityLevel, ValidationError


@dataclass
class ThresholdConfig:
    """Configuration for spike detection thresholds."""
    
    # Multiplier thresholds for different severity levels
    low_threshold: float = 1.5      # 50% above baseline
    medium_threshold: float = 2.0   # 100% above baseline  
    high_threshold: float = 3.0     # 200% above baseline
    critical_threshold: float = 5.0 # 400% above baseline
    
    # Minimum baseline latency to consider (ms)
    min_baseline_latency: float = 10.0
    
    # Time window for calculating baseline (minutes)
    baseline_window_minutes: int = 15
    
    # Minimum number of data points for baseline calculation
    min_baseline_points: int = 5
    
    # Aggregation window for preventing alert fatigue (minutes)
    aggregation_window_minutes: int = 5
    
    def validate(self) -> None:
        """Validate threshold configuration."""
        thresholds = [self.low_threshold, self.medium_threshold, 
                     self.high_threshold, self.critical_threshold]
        
        # Check that thresholds are in ascending order
        if not all(thresholds[i] < thresholds[i+1] for i in range(len(thresholds)-1)):
            raise ValidationError("Thresholds must be in ascending order")
        
        if any(t <= 1.0 for t in thresholds):
            raise ValidationError("All thresholds must be greater than 1.0")
        
        if self.min_baseline_latency <= 0:
            raise ValidationError("min_baseline_latency must be positive")
        
        if self.baseline_window_minutes <= 0:
            raise ValidationError("baseline_window_minutes must be positive")
        
        if self.min_baseline_points <= 0:
            raise ValidationError("min_baseline_points must be positive")
        
        if self.aggregation_window_minutes <= 0:
            raise ValidationError("aggregation_window_minutes must be positive")


@dataclass
class SystemStatus:
    """Current system status for monitoring."""
    
    is_healthy: bool = True
    active_spikes: int = 0
    last_check_time: Optional[datetime] = None
    baseline_latencies: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None


class SpikeDetector:
    """
    Detects latency spikes in metric data using configurable thresholds.
    
    Features:
    - Configurable threshold-based detection with multiple severity levels
    - Baseline calculation using historical data
    - Spike aggregation to prevent alert fatigue
    - Automatic severity level calculation
    """
    
    def __init__(self, config: Optional[ThresholdConfig] = None):
        """Initialize the spike detector with configuration."""
        self.config = config or ThresholdConfig()
        self.config.validate()
        
        # Storage for historical metrics and active spikes
        self._metric_history: Dict[str, List[Metric]] = {}
        self._active_spikes: Dict[str, SpikeEvent] = {}
        self._baseline_cache: Dict[str, Tuple[float, datetime]] = {}
        
        # System status
        self._status = SystemStatus()
    
    def detect_spikes(self, metrics: List[Metric]) -> List[SpikeEvent]:
        """
        Detect spikes in the provided metrics.
        
        Args:
            metrics: List of metric data points to analyze
            
        Returns:
            List of detected spike events
            
        Raises:
            ValidationError: If metrics are invalid
        """
        if not metrics:
            return []
        
        # Validate all metrics
        for metric in metrics:
            metric.validate()
        
        detected_spikes = []
        current_time = datetime.now()
        
        try:
            # Update metric history
            self._update_metric_history(metrics)
            
            # Group metrics by endpoint/source for analysis
            metrics_by_endpoint = self._group_metrics_by_endpoint(metrics)
            
            for endpoint, endpoint_metrics in metrics_by_endpoint.items():
                # Calculate baseline for this endpoint
                baseline = self._calculate_baseline(endpoint, current_time)
                
                if baseline is None:
                    continue  # Not enough data for baseline
                
                # Check each metric for spikes
                for metric in endpoint_metrics:
                    spike = self._check_for_spike(metric, baseline, endpoint)
                    if spike:
                        # Check if this should be aggregated with existing spike
                        aggregated_spike = self._aggregate_spike(spike)
                        if aggregated_spike:
                            detected_spikes.append(aggregated_spike)
            
            # Update system status
            self._update_system_status(detected_spikes, current_time)
            
        except Exception as e:
            self._status.is_healthy = False
            self._status.error_message = str(e)
            raise
        
        return detected_spikes
    
    def configure_thresholds(self, thresholds: Dict[str, float]) -> None:
        """
        Update threshold configuration.
        
        Args:
            thresholds: Dictionary of threshold names to values
        """
        # Update config with provided thresholds
        for key, value in thresholds.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Validate updated configuration
        self.config.validate()
        
        # Clear baseline cache to force recalculation
        self._baseline_cache.clear()
    
    def get_current_status(self) -> SystemStatus:
        """Get current system status."""
        return self._status
    
    def _update_metric_history(self, metrics: List[Metric]) -> None:
        """Update the internal metric history with new data points."""
        cutoff_time = datetime.now() - timedelta(
            minutes=self.config.baseline_window_minutes * 2
        )
        
        for metric in metrics:
            endpoint_key = f"{metric.source}:{metric.metric_name}"
            
            if endpoint_key not in self._metric_history:
                self._metric_history[endpoint_key] = []
            
            # Add new metric
            self._metric_history[endpoint_key].append(metric)
            
            # Remove old metrics beyond the retention window
            self._metric_history[endpoint_key] = [
                m for m in self._metric_history[endpoint_key]
                if m.timestamp > cutoff_time
            ]
            
            # Sort by timestamp
            self._metric_history[endpoint_key].sort(key=lambda m: m.timestamp)
    
    def _group_metrics_by_endpoint(self, metrics: List[Metric]) -> Dict[str, List[Metric]]:
        """Group metrics by endpoint for analysis."""
        grouped = {}
        for metric in metrics:
            endpoint_key = f"{metric.source}:{metric.metric_name}"
            if endpoint_key not in grouped:
                grouped[endpoint_key] = []
            grouped[endpoint_key].append(metric)
        return grouped
    
    def _calculate_baseline(self, endpoint: str, current_time: datetime) -> Optional[float]:
        """
        Calculate baseline latency for an endpoint.
        
        Args:
            endpoint: Endpoint identifier
            current_time: Current timestamp for cache validation
            
        Returns:
            Baseline latency value or None if insufficient data
        """
        # Check cache first
        if endpoint in self._baseline_cache:
            baseline, cache_time = self._baseline_cache[endpoint]
            cache_age = (current_time - cache_time).total_seconds() / 60
            if cache_age < 5:  # Cache valid for 5 minutes
                return baseline
        
        # Get historical data for baseline calculation
        if endpoint not in self._metric_history:
            return None
        
        history = self._metric_history[endpoint]
        baseline_cutoff = current_time - timedelta(
            minutes=self.config.baseline_window_minutes
        )
        
        # Filter to baseline window
        baseline_metrics = [
            m for m in history 
            if baseline_cutoff <= m.timestamp < current_time
        ]
        
        if len(baseline_metrics) < self.config.min_baseline_points:
            return None
        
        # Calculate baseline using median (more robust than mean)
        baseline_values = [m.value for m in baseline_metrics]
        baseline = statistics.median(baseline_values)
        
        # Cache the result
        self._baseline_cache[endpoint] = (baseline, current_time)
        
        return baseline
    
    def _check_for_spike(self, metric: Metric, baseline: float, endpoint: str) -> Optional[SpikeEvent]:
        """
        Check if a metric represents a spike.
        
        Args:
            metric: Metric to check
            baseline: Baseline latency for comparison
            endpoint: Endpoint identifier
            
        Returns:
            SpikeEvent if spike detected, None otherwise
        """
        # Skip if baseline is too low to be meaningful
        if baseline < self.config.min_baseline_latency:
            return None
        
        # Calculate spike ratio
        spike_ratio = metric.value / baseline
        
        # Determine severity level
        severity = self._calculate_severity(spike_ratio)
        
        if severity is None:
            return None  # No spike detected
        
        # Create spike event
        spike = SpikeEvent(
            timestamp=metric.timestamp,
            endpoint=endpoint,
            severity=severity,
            baseline_latency=baseline,
            spike_latency=metric.value,
            duration=timedelta(seconds=0),  # Will be updated during aggregation
            affected_metrics=[metric.metric_name],
            metadata={
                'spike_ratio': spike_ratio,
                'source': metric.source,
                'detection_time': datetime.now().isoformat()
            }
        )
        
        return spike
    
    def _calculate_severity(self, spike_ratio: float) -> Optional[SeverityLevel]:
        """
        Calculate severity level based on spike ratio.
        
        Args:
            spike_ratio: Ratio of current value to baseline
            
        Returns:
            SeverityLevel or None if no spike
        """
        if spike_ratio >= self.config.critical_threshold:
            return SeverityLevel.CRITICAL
        elif spike_ratio >= self.config.high_threshold:
            return SeverityLevel.HIGH
        elif spike_ratio >= self.config.medium_threshold:
            return SeverityLevel.MEDIUM
        elif spike_ratio >= self.config.low_threshold:
            return SeverityLevel.LOW
        else:
            return None
    
    def _aggregate_spike(self, new_spike: SpikeEvent) -> Optional[SpikeEvent]:
        """
        Aggregate spike with existing active spikes to prevent alert fatigue.
        
        Args:
            new_spike: Newly detected spike
            
        Returns:
            Aggregated spike event or None if aggregated with existing
        """
        endpoint = new_spike.endpoint
        aggregation_window = timedelta(minutes=self.config.aggregation_window_minutes)
        
        # Check if there's an active spike for this endpoint
        if endpoint in self._active_spikes:
            active_spike = self._active_spikes[endpoint]
            time_diff = new_spike.timestamp - active_spike.timestamp
            
            # If within aggregation window, update existing spike
            if time_diff <= aggregation_window:
                # Update the active spike with new information
                active_spike.duration = time_diff
                active_spike.spike_latency = max(active_spike.spike_latency, new_spike.spike_latency)
                
                # Update severity to the highest level
                if new_spike.severity.value != active_spike.severity.value:
                    severity_order = [SeverityLevel.LOW, SeverityLevel.MEDIUM, 
                                    SeverityLevel.HIGH, SeverityLevel.CRITICAL]
                    if severity_order.index(new_spike.severity) > severity_order.index(active_spike.severity):
                        active_spike.severity = new_spike.severity
                
                # Merge affected metrics
                for metric in new_spike.affected_metrics:
                    if metric not in active_spike.affected_metrics:
                        active_spike.affected_metrics.append(metric)
                
                return None  # Spike was aggregated, don't return new event
        
        # This is a new spike or outside aggregation window
        self._active_spikes[endpoint] = new_spike
        return new_spike
    
    def _update_system_status(self, detected_spikes: List[SpikeEvent], current_time: datetime) -> None:
        """Update system status based on detection results."""
        self._status.last_check_time = current_time
        self._status.active_spikes = len(self._active_spikes)
        self._status.is_healthy = True
        self._status.error_message = None
        
        # Update baseline latencies in status
        self._status.baseline_latencies = {
            endpoint: baseline for endpoint, (baseline, _) in self._baseline_cache.items()
        }
        
        # Clean up old active spikes
        cleanup_cutoff = current_time - timedelta(minutes=self.config.aggregation_window_minutes * 2)
        expired_endpoints = [
            endpoint for endpoint, spike in self._active_spikes.items()
            if spike.timestamp < cleanup_cutoff
        ]
        
        for endpoint in expired_endpoints:
            del self._active_spikes[endpoint]