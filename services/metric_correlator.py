"""
Metric Correlation System for the Latency Spike Root Cause Investigator.

This module implements the MetricCorrelator class that retrieves metrics from multiple
sources within time windows and calculates correlation scores to identify relationships
between latency spikes and other system metrics.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import statistics
import math

from models import Metric, SpikeEvent, CorrelationResult
from clients.manager import MCPClientManager


logger = logging.getLogger(__name__)


@dataclass
class MetricData:
    """Container for time-series metric data."""
    metrics: List[Metric] = field(default_factory=list)
    source: str = ""
    metric_type: str = ""
    
    def get_values(self) -> List[float]:
        """Extract just the values from metrics."""
        return [m.value for m in self.metrics]
    
    def get_timestamps(self) -> List[datetime]:
        """Extract just the timestamps from metrics."""
        return [m.timestamp for m in self.metrics]
    
    def is_empty(self) -> bool:
        """Check if metric data is empty."""
        return len(self.metrics) == 0
    
    def get_time_range(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get the time range covered by this metric data."""
        if self.is_empty():
            return None, None
        timestamps = self.get_timestamps()
        return min(timestamps), max(timestamps)


@dataclass
class CorrelationConfig:
    """Configuration for correlation analysis."""
    time_window_minutes: int = 15  # Time window around spike to analyze
    min_data_points: int = 5  # Minimum data points required for correlation
    correlation_threshold: float = 0.3  # Minimum correlation to consider significant
    timeout_seconds: float = 30.0  # Timeout for data retrieval
    interpolate_missing: bool = True  # Whether to interpolate missing data points


class MetricCorrelator:
    """
    Correlates latency spikes with network and database metrics to identify
    potential root causes through statistical correlation analysis.
    """
    
    def __init__(self, client_manager: MCPClientManager, config: Optional[CorrelationConfig] = None):
        self.client_manager = client_manager
        self.config = config or CorrelationConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def correlate_metrics(self, spike: SpikeEvent) -> CorrelationResult:
        """
        Correlate a spike event with network and database metrics.
        
        Args:
            spike: The spike event to analyze
            
        Returns:
            CorrelationResult with correlation scores and confidence
        """
        self.logger.info(f"Starting correlation analysis for spike at {spike.timestamp}")
        
        # Calculate time window around the spike
        start_time, end_time = self._calculate_time_window(spike.timestamp)
        
        # Retrieve metrics from all sources concurrently
        try:
            network_data, db_data = await asyncio.wait_for(
                self._retrieve_all_metrics(start_time, end_time),
                timeout=self.config.timeout_seconds
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout retrieving metrics for spike at {spike.timestamp}")
            network_data, db_data = MetricData(), MetricData()
        except Exception as e:
            self.logger.error(f"Error retrieving metrics: {e}")
            network_data, db_data = MetricData(), MetricData()
        
        # Calculate correlation scores
        correlation_scores = await self._calculate_correlation_scores(
            spike, network_data, db_data, start_time, end_time
        )
        
        # Calculate overall confidence based on data availability and correlation strength
        confidence = self._calculate_confidence(network_data, db_data, correlation_scores)
        
        # Create correlation result
        result = CorrelationResult(
            spike_event=spike,
            network_metrics=self._extract_metric_summary(network_data),
            db_metrics=self._extract_metric_summary(db_data),
            correlation_scores=correlation_scores,
            confidence=confidence,
            analysis_timestamp=datetime.now()
        )
        
        self.logger.info(
            f"Correlation analysis complete. Found {len(correlation_scores)} correlations "
            f"with confidence {confidence:.2f}"
        )
        
        return result
    
    async def _retrieve_all_metrics(
        self, start_time: datetime, end_time: datetime
    ) -> Tuple[MetricData, MetricData]:
        """Retrieve metrics from all available sources concurrently."""
        tasks = []
        
        # Network metrics
        network_task = asyncio.create_task(
            self._get_network_metrics(start_time, end_time)
        )
        tasks.append(network_task)
        
        # Database metrics
        db_task = asyncio.create_task(
            self._get_database_metrics(start_time, end_time)
        )
        tasks.append(db_task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Extract results, handling exceptions
        network_data = results[0] if not isinstance(results[0], Exception) else MetricData()
        db_data = results[1] if not isinstance(results[1], Exception) else MetricData()
        
        if isinstance(results[0], Exception):
            self.logger.error(f"Error retrieving network metrics: {results[0]}")
        if isinstance(results[1], Exception):
            self.logger.error(f"Error retrieving database metrics: {results[1]}")
        
        return network_data, db_data
    
    async def _get_network_metrics(self, start_time: datetime, end_time: datetime) -> MetricData:
        """Retrieve network metrics for the specified time window."""
        network_client = self.client_manager.get_network_client()
        if not network_client:
            self.logger.warning("No network client available")
            return MetricData()
        
        try:
            # Get various network metrics
            bandwidth_metrics = await network_client.get_bandwidth_metrics(start_time, end_time)
            latency_metrics = await network_client.get_latency_metrics(start_time, end_time)
            packet_loss_metrics = await network_client.get_packet_loss_metrics(start_time, end_time)
            connection_metrics = await network_client.get_connection_count_metrics(start_time, end_time)
            
            # Combine all network metrics
            all_metrics = bandwidth_metrics + latency_metrics + packet_loss_metrics + connection_metrics
            
            return MetricData(
                metrics=all_metrics,
                source="network",
                metric_type="combined"
            )
        except Exception as e:
            self.logger.error(f"Error retrieving network metrics: {e}")
            return MetricData()
    
    async def _get_database_metrics(self, start_time: datetime, end_time: datetime) -> MetricData:
        """Retrieve database metrics for the specified time window."""
        db_client = self.client_manager.get_database_client()
        if not db_client:
            self.logger.warning("No database client available")
            return MetricData()
        
        try:
            # Get various database metrics
            query_metrics = await db_client.get_query_performance_metrics(start_time, end_time)
            connection_metrics = await db_client.get_connection_pool_metrics(start_time, end_time)
            lock_metrics = await db_client.get_lock_metrics(start_time, end_time)
            
            # Combine all database metrics
            all_metrics = query_metrics + connection_metrics + lock_metrics
            
            return MetricData(
                metrics=all_metrics,
                source="database",
                metric_type="combined"
            )
        except Exception as e:
            self.logger.error(f"Error retrieving database metrics: {e}")
            return MetricData()
    
    async def get_time_window_data(
        self, start_time: datetime, end_time: datetime
    ) -> Dict[str, MetricData]:
        """
        Get metric data for a specific time window from all sources.
        
        Args:
            start_time: Start of the time window
            end_time: End of the time window
            
        Returns:
            Dictionary mapping source names to MetricData
        """
        self.logger.debug(f"Retrieving time window data from {start_time} to {end_time}")
        
        try:
            network_data, db_data = await asyncio.wait_for(
                self._retrieve_all_metrics(start_time, end_time),
                timeout=self.config.timeout_seconds
            )
            
            return {
                "network": network_data,
                "database": db_data
            }
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout retrieving time window data")
            return {"network": MetricData(), "database": MetricData()}
        except Exception as e:
            self.logger.error(f"Error retrieving time window data: {e}")
            return {"network": MetricData(), "database": MetricData()}
    
    async def _calculate_correlation_scores(
        self,
        spike: SpikeEvent,
        network_data: MetricData,
        db_data: MetricData,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, float]:
        """Calculate correlation scores between spike and other metrics."""
        correlation_scores = {}
        
        # Get baseline latency data for comparison
        baseline_data = await self._get_baseline_latency_data(spike, start_time, end_time)
        
        if baseline_data.is_empty():
            self.logger.warning("No baseline latency data available for correlation")
            return correlation_scores
        
        # Calculate correlations with network metrics
        network_correlations = self._calculate_metric_correlations(
            baseline_data, network_data, "network"
        )
        correlation_scores.update(network_correlations)
        
        # Calculate correlations with database metrics
        db_correlations = self._calculate_metric_correlations(
            baseline_data, db_data, "database"
        )
        correlation_scores.update(db_correlations)
        
        # Filter out weak correlations
        significant_correlations = {
            metric: score for metric, score in correlation_scores.items()
            if abs(score) >= self.config.correlation_threshold
        }
        
        self.logger.debug(
            f"Calculated {len(correlation_scores)} total correlations, "
            f"{len(significant_correlations)} significant"
        )
        
        return significant_correlations
    
    def _calculate_metric_correlations(
        self, baseline_data: MetricData, metric_data: MetricData, source_prefix: str
    ) -> Dict[str, float]:
        """Calculate correlations between baseline and metric data."""
        correlations = {}
        
        if metric_data.is_empty():
            return correlations
        
        # Group metrics by name
        metric_groups = self._group_metrics_by_name(metric_data.metrics)
        baseline_values = baseline_data.get_values()
        baseline_timestamps = baseline_data.get_timestamps()
        
        for metric_name, metrics in metric_groups.items():
            try:
                # Align metric data with baseline timestamps
                aligned_values = self._align_metric_data(
                    metrics, baseline_timestamps
                )
                
                if len(aligned_values) < self.config.min_data_points:
                    continue
                
                # Calculate Pearson correlation
                correlation = self._calculate_pearson_correlation(
                    baseline_values[:len(aligned_values)], aligned_values
                )
                
                if not math.isnan(correlation):
                    correlations[f"{source_prefix}_{metric_name}"] = correlation
                    
            except Exception as e:
                self.logger.debug(f"Error calculating correlation for {metric_name}: {e}")
                continue
        
        return correlations
    
    def _group_metrics_by_name(self, metrics: List[Metric]) -> Dict[str, List[Metric]]:
        """Group metrics by their metric_name."""
        groups = {}
        for metric in metrics:
            if metric.metric_name not in groups:
                groups[metric.metric_name] = []
            groups[metric.metric_name].append(metric)
        
        # Sort each group by timestamp
        for metric_name in groups:
            groups[metric_name].sort(key=lambda m: m.timestamp)
        
        return groups
    
    def _align_metric_data(
        self, metrics: List[Metric], target_timestamps: List[datetime]
    ) -> List[float]:
        """Align metric data to target timestamps, interpolating if necessary."""
        if not metrics or not target_timestamps:
            return []
        
        # Sort metrics by timestamp
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
        aligned_values = []
        used_metrics = set()  # Track which metrics we've already used
        
        for target_time in target_timestamps:
            # Find the closest unused metric value
            available_metrics = [m for i, m in enumerate(sorted_metrics) if i not in used_metrics]
            
            if not available_metrics:
                # No more metrics available, use interpolation if enabled
                if self.config.interpolate_missing and len(aligned_values) > 0:
                    aligned_values.append(aligned_values[-1])
                continue
            
            closest_metric = min(
                available_metrics,
                key=lambda m: abs((m.timestamp - target_time).total_seconds())
            )
            
            # Use the closest value if within reasonable time tolerance (5 minutes)
            time_diff = abs((closest_metric.timestamp - target_time).total_seconds())
            if time_diff <= 300:  # 5 minutes
                aligned_values.append(closest_metric.value)
                # Mark this metric as used
                metric_index = sorted_metrics.index(closest_metric)
                used_metrics.add(metric_index)
            elif self.config.interpolate_missing and len(aligned_values) > 0:
                # Simple interpolation using the last known value
                aligned_values.append(aligned_values[-1])
            else:
                # Skip this data point
                continue
        
        return aligned_values
    
    def _calculate_pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient between two datasets."""
        if len(x) != len(y) or len(x) < 2:
            return float('nan')
        
        try:
            # Calculate means
            mean_x = statistics.mean(x)
            mean_y = statistics.mean(y)
            
            # Calculate correlation coefficient
            numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
            
            sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
            sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(len(y)))
            
            denominator = math.sqrt(sum_sq_x * sum_sq_y)
            
            if denominator == 0:
                return float('nan')
            
            return numerator / denominator
            
        except Exception as e:
            self.logger.debug(f"Error calculating Pearson correlation: {e}")
            return float('nan')
    
    async def _get_baseline_latency_data(
        self, spike: SpikeEvent, start_time: datetime, end_time: datetime
    ) -> MetricData:
        """Get latency data for the time window to use as baseline for correlation."""
        apm_client = self.client_manager.get_apm_client()
        if not apm_client:
            self.logger.warning("No APM client available for baseline data")
            return MetricData()
        
        try:
            latency_metrics = await apm_client.get_latency_metrics(
                start_time, end_time, spike.endpoint
            )
            
            return MetricData(
                metrics=latency_metrics,
                source="apm",
                metric_type="latency"
            )
        except Exception as e:
            self.logger.error(f"Error retrieving baseline latency data: {e}")
            return MetricData()
    
    def _calculate_time_window(self, spike_time: datetime) -> Tuple[datetime, datetime]:
        """Calculate the time window around a spike for correlation analysis."""
        window_delta = timedelta(minutes=self.config.time_window_minutes)
        start_time = spike_time - window_delta
        end_time = spike_time + window_delta
        return start_time, end_time
    
    def _extract_metric_summary(self, metric_data: MetricData) -> Dict[str, float]:
        """Extract summary statistics from metric data."""
        if metric_data.is_empty():
            return {}
        
        # Group metrics by name and calculate summary stats
        metric_groups = self._group_metrics_by_name(metric_data.metrics)
        summary = {}
        
        for metric_name, metrics in metric_groups.items():
            values = [m.value for m in metrics]
            if values:
                summary[f"{metric_name}_avg"] = statistics.mean(values)
                summary[f"{metric_name}_max"] = max(values)
                summary[f"{metric_name}_min"] = min(values)
                if len(values) > 1:
                    summary[f"{metric_name}_stddev"] = statistics.stdev(values)
        
        return summary
    
    def _calculate_confidence(
        self,
        network_data: MetricData,
        db_data: MetricData,
        correlation_scores: Dict[str, float]
    ) -> float:
        """Calculate confidence score based on data availability and correlation strength."""
        confidence_factors = []
        
        # Data availability factor (0.0 to 0.4)
        data_availability = 0.0
        if not network_data.is_empty():
            data_availability += 0.2
        if not db_data.is_empty():
            data_availability += 0.2
        confidence_factors.append(data_availability)
        
        # Correlation strength factor (0.0 to 0.4)
        if correlation_scores:
            max_correlation = max(abs(score) for score in correlation_scores.values())
            correlation_strength = min(max_correlation, 1.0) * 0.4
            confidence_factors.append(correlation_strength)
        else:
            confidence_factors.append(0.0)
        
        # Number of correlations factor (0.0 to 0.2)
        num_correlations = len(correlation_scores)
        correlation_count_factor = min(num_correlations / 10.0, 1.0) * 0.2
        confidence_factors.append(correlation_count_factor)
        
        # Calculate overall confidence
        total_confidence = sum(confidence_factors)
        return min(total_confidence, 1.0)
    
    def calculate_correlation_scores(
        self, metrics: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Calculate correlation scores between different metric series.
        
        Args:
            metrics: Dictionary mapping metric names to value lists
            
        Returns:
            Dictionary mapping metric pairs to correlation scores
        """
        correlation_scores = {}
        metric_names = list(metrics.keys())
        
        # Calculate pairwise correlations
        for i in range(len(metric_names)):
            for j in range(i + 1, len(metric_names)):
                metric1 = metric_names[i]
                metric2 = metric_names[j]
                
                values1 = metrics[metric1]
                values2 = metrics[metric2]
                
                # Ensure both series have the same length
                min_length = min(len(values1), len(values2))
                if min_length < self.config.min_data_points:
                    continue
                
                correlation = self._calculate_pearson_correlation(
                    values1[:min_length], values2[:min_length]
                )
                
                if not math.isnan(correlation):
                    pair_name = f"{metric1}_vs_{metric2}"
                    correlation_scores[pair_name] = correlation
        
        return correlation_scores