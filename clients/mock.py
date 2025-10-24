"""
Mock implementations of MCP clients for development and testing.

These mock clients return realistic test data to simulate real monitoring systems
without requiring actual external connections.
"""

import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import math

from models import Metric, SpikeEvent, CorrelationResult, SeverityLevel
from .base import (
    APMClient, NetworkClient, DatabaseClient, KnowledgeBaseClient,
    HealthCheckResult, ConnectionStatus, HistoricalIncident
)


class MockAPMClient(APMClient):
    """Mock APM client that generates realistic latency and performance data."""
    
    def __init__(self, name: str = "mock_apm", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config or {})
        self._endpoints = [
            "/api/users", "/api/orders", "/api/products", "/api/auth/login",
            "/api/payments", "/api/inventory", "/api/reports", "/api/notifications"
        ]
        self._baseline_latencies = {
            "/api/users": 120.0,
            "/api/orders": 250.0,
            "/api/products": 80.0,
            "/api/auth/login": 300.0,
            "/api/payments": 450.0,
            "/api/inventory": 180.0,
            "/api/reports": 800.0,
            "/api/notifications": 90.0
        }
    
    async def connect(self) -> bool:
        """Simulate connection to APM service."""
        await asyncio.sleep(0.1)  # Simulate connection delay
        self._update_connection_status(ConnectionStatus.CONNECTED)
        return True
    
    async def disconnect(self) -> None:
        """Simulate disconnection from APM service."""
        await asyncio.sleep(0.05)
        self._update_connection_status(ConnectionStatus.DISCONNECTED)
    
    async def health_check(self) -> HealthCheckResult:
        """Perform mock health check."""
        start_time = datetime.now()
        await asyncio.sleep(0.02)  # Simulate API call
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Simulate occasional connection issues
        if random.random() < 0.05:  # 5% chance of failure
            result = HealthCheckResult(
                status=ConnectionStatus.ERROR,
                response_time_ms=response_time,
                error_message="Connection timeout"
            )
            self._update_connection_status(ConnectionStatus.ERROR)
        else:
            result = HealthCheckResult(
                status=ConnectionStatus.CONNECTED,
                response_time_ms=response_time
            )
            self._update_connection_status(ConnectionStatus.CONNECTED)
        
        self._last_health_check = result
        return result
    
    async def get_latency_metrics(
        self, 
        start_time: datetime, 
        end_time: datetime,
        endpoint: Optional[str] = None
    ) -> List[Metric]:
        """Generate realistic latency metrics."""
        metrics = []
        duration = end_time - start_time
        interval = timedelta(minutes=1)  # 1-minute intervals
        
        endpoints = [endpoint] if endpoint else self._endpoints
        
        current_time = start_time
        while current_time <= end_time:
            for ep in endpoints:
                baseline = self._baseline_latencies.get(ep, 200.0)
                
                # Add some realistic variation
                noise = random.gauss(0, baseline * 0.1)  # 10% standard deviation
                
                # Simulate occasional spikes
                if random.random() < 0.02:  # 2% chance of spike
                    spike_multiplier = random.uniform(2.0, 5.0)
                    value = baseline * spike_multiplier + noise
                else:
                    value = baseline + noise
                
                # Ensure positive values
                value = max(value, 10.0)
                
                metric = Metric(
                    timestamp=current_time,
                    value=value,
                    metric_name="response_time_ms",
                    source=f"apm_mock_{ep}",
                    metadata={
                        "endpoint": ep,
                        "method": "GET",
                        "status_code": 200
                    }
                )
                metrics.append(metric)
            
            current_time += interval
        
        return metrics
    
    async def get_throughput_metrics(
        self, 
        start_time: datetime, 
        end_time: datetime,
        endpoint: Optional[str] = None
    ) -> List[Metric]:
        """Generate realistic throughput metrics."""
        metrics = []
        duration = end_time - start_time
        interval = timedelta(minutes=1)
        
        endpoints = [endpoint] if endpoint else self._endpoints
        
        current_time = start_time
        while current_time <= end_time:
            for ep in endpoints:
                # Base throughput varies by endpoint
                base_throughput = {
                    "/api/users": 50.0,
                    "/api/orders": 30.0,
                    "/api/products": 80.0,
                    "/api/auth/login": 20.0,
                    "/api/payments": 15.0,
                    "/api/inventory": 40.0,
                    "/api/reports": 5.0,
                    "/api/notifications": 100.0
                }.get(ep, 25.0)
                
                # Add time-of-day variation
                hour = current_time.hour
                if 9 <= hour <= 17:  # Business hours
                    multiplier = 1.5
                elif 18 <= hour <= 22:  # Evening
                    multiplier = 1.2
                else:  # Night/early morning
                    multiplier = 0.3
                
                value = base_throughput * multiplier + random.gauss(0, base_throughput * 0.2)
                value = max(value, 0.0)
                
                metric = Metric(
                    timestamp=current_time,
                    value=value,
                    metric_name="requests_per_minute",
                    source=f"apm_mock_{ep}",
                    metadata={"endpoint": ep}
                )
                metrics.append(metric)
            
            current_time += interval
        
        return metrics
    
    async def get_error_rate_metrics(
        self, 
        start_time: datetime, 
        end_time: datetime,
        endpoint: Optional[str] = None
    ) -> List[Metric]:
        """Generate realistic error rate metrics."""
        metrics = []
        interval = timedelta(minutes=1)
        
        endpoints = [endpoint] if endpoint else self._endpoints
        
        current_time = start_time
        while current_time <= end_time:
            for ep in endpoints:
                # Most endpoints have low error rates
                base_error_rate = 0.5  # 0.5% base error rate
                
                # Some endpoints are more error-prone
                if ep in ["/api/payments", "/api/auth/login"]:
                    base_error_rate = 1.5
                
                # Simulate occasional error spikes
                if random.random() < 0.01:  # 1% chance of error spike
                    value = base_error_rate * random.uniform(5.0, 20.0)
                else:
                    value = base_error_rate + random.gauss(0, 0.2)
                
                value = max(value, 0.0)
                
                metric = Metric(
                    timestamp=current_time,
                    value=value,
                    metric_name="error_rate_percent",
                    source=f"apm_mock_{ep}",
                    metadata={"endpoint": ep}
                )
                metrics.append(metric)
            
            current_time += interval
        
        return metrics
    
    async def get_available_endpoints(self) -> List[str]:
        """Return list of mock endpoints."""
        return self._endpoints.copy()


class MockNetworkClient(NetworkClient):
    """Mock Network client that generates realistic network metrics."""
    
    def __init__(self, name: str = "mock_network", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config or {})
    
    async def connect(self) -> bool:
        """Simulate connection to network monitoring service."""
        await asyncio.sleep(0.1)
        self._update_connection_status(ConnectionStatus.CONNECTED)
        return True
    
    async def disconnect(self) -> None:
        """Simulate disconnection from network monitoring service."""
        await asyncio.sleep(0.05)
        self._update_connection_status(ConnectionStatus.DISCONNECTED)
    
    async def health_check(self) -> HealthCheckResult:
        """Perform mock health check."""
        start_time = datetime.now()
        await asyncio.sleep(0.03)
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        
        result = HealthCheckResult(
            status=ConnectionStatus.CONNECTED,
            response_time_ms=response_time
        )
        self._update_connection_status(ConnectionStatus.CONNECTED)
        self._last_health_check = result
        return result
    
    async def get_bandwidth_metrics(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[Metric]:
        """Generate realistic bandwidth metrics."""
        metrics = []
        interval = timedelta(minutes=1)
        
        current_time = start_time
        while current_time <= end_time:
            # Simulate bandwidth utilization (0-100%)
            base_utilization = 45.0  # 45% average utilization
            
            # Add time-based variation
            hour = current_time.hour
            if 9 <= hour <= 17:  # Business hours
                multiplier = 1.3
            else:
                multiplier = 0.7
            
            value = base_utilization * multiplier + random.gauss(0, 10.0)
            value = max(0.0, min(100.0, value))  # Clamp between 0-100%
            
            metric = Metric(
                timestamp=current_time,
                value=value,
                metric_name="bandwidth_utilization_percent",
                source="network_mock",
                metadata={"interface": "eth0"}
            )
            metrics.append(metric)
            
            current_time += interval
        
        return metrics
    
    async def get_latency_metrics(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[Metric]:
        """Generate realistic network latency metrics."""
        metrics = []
        interval = timedelta(minutes=1)
        
        current_time = start_time
        while current_time <= end_time:
            # Base network latency around 2ms
            base_latency = 2.0
            
            # Add realistic variation
            value = base_latency + random.gauss(0, 0.5)
            
            # Simulate occasional network congestion
            if random.random() < 0.03:  # 3% chance
                value += random.uniform(5.0, 20.0)
            
            value = max(value, 0.1)
            
            metric = Metric(
                timestamp=current_time,
                value=value,
                metric_name="network_latency_ms",
                source="network_mock",
                metadata={"target": "gateway"}
            )
            metrics.append(metric)
            
            current_time += interval
        
        return metrics
    
    async def get_packet_loss_metrics(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[Metric]:
        """Generate realistic packet loss metrics."""
        metrics = []
        interval = timedelta(minutes=1)
        
        current_time = start_time
        while current_time <= end_time:
            # Very low packet loss normally
            base_loss = 0.01  # 0.01%
            
            # Simulate occasional packet loss spikes
            if random.random() < 0.02:  # 2% chance
                value = random.uniform(0.5, 3.0)
            else:
                value = base_loss + random.gauss(0, 0.005)
            
            value = max(value, 0.0)
            
            metric = Metric(
                timestamp=current_time,
                value=value,
                metric_name="packet_loss_percent",
                source="network_mock",
                metadata={"interface": "eth0"}
            )
            metrics.append(metric)
            
            current_time += interval
        
        return metrics
    
    async def get_connection_count_metrics(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[Metric]:
        """Generate realistic connection count metrics."""
        metrics = []
        interval = timedelta(minutes=1)
        
        current_time = start_time
        while current_time <= end_time:
            # Base connection count
            base_connections = 150
            
            # Add time-based variation
            hour = current_time.hour
            if 9 <= hour <= 17:  # Business hours
                multiplier = 1.5
            else:
                multiplier = 0.6
            
            value = base_connections * multiplier + random.gauss(0, 20)
            value = max(value, 0)
            
            metric = Metric(
                timestamp=current_time,
                value=int(value),
                metric_name="active_connections",
                source="network_mock",
                metadata={"protocol": "tcp"}
            )
            metrics.append(metric)
            
            current_time += interval
        
        return metrics


class MockDatabaseClient(DatabaseClient):
    """Mock Database client that generates realistic database metrics."""
    
    def __init__(self, name: str = "mock_database", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config or {})
    
    async def connect(self) -> bool:
        """Simulate connection to database monitoring service."""
        await asyncio.sleep(0.1)
        self._update_connection_status(ConnectionStatus.CONNECTED)
        return True
    
    async def disconnect(self) -> None:
        """Simulate disconnection from database monitoring service."""
        await asyncio.sleep(0.05)
        self._update_connection_status(ConnectionStatus.DISCONNECTED)
    
    async def health_check(self) -> HealthCheckResult:
        """Perform mock health check."""
        start_time = datetime.now()
        await asyncio.sleep(0.04)
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        
        result = HealthCheckResult(
            status=ConnectionStatus.CONNECTED,
            response_time_ms=response_time
        )
        self._update_connection_status(ConnectionStatus.CONNECTED)
        self._last_health_check = result
        return result
    
    async def get_query_performance_metrics(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[Metric]:
        """Generate realistic query performance metrics."""
        metrics = []
        interval = timedelta(minutes=1)
        
        current_time = start_time
        while current_time <= end_time:
            # Average query time
            base_query_time = 25.0  # 25ms average
            
            # Add realistic variation
            value = base_query_time + random.gauss(0, 8.0)
            
            # Simulate occasional slow queries
            if random.random() < 0.05:  # 5% chance
                value += random.uniform(100.0, 500.0)
            
            value = max(value, 1.0)
            
            metric = Metric(
                timestamp=current_time,
                value=value,
                metric_name="avg_query_time_ms",
                source="database_mock",
                metadata={"database": "main_db"}
            )
            metrics.append(metric)
            
            current_time += interval
        
        return metrics
    
    async def get_connection_pool_metrics(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[Metric]:
        """Generate realistic connection pool metrics."""
        metrics = []
        interval = timedelta(minutes=1)
        
        current_time = start_time
        while current_time <= end_time:
            # Connection pool utilization
            base_utilization = 60.0  # 60% average utilization
            
            # Add time-based variation
            hour = current_time.hour
            if 9 <= hour <= 17:  # Business hours
                multiplier = 1.2
            else:
                multiplier = 0.8
            
            value = base_utilization * multiplier + random.gauss(0, 10.0)
            value = max(0.0, min(100.0, value))
            
            metric = Metric(
                timestamp=current_time,
                value=value,
                metric_name="connection_pool_utilization_percent",
                source="database_mock",
                metadata={"pool": "main_pool", "max_connections": 100}
            )
            metrics.append(metric)
            
            current_time += interval
        
        return metrics
    
    async def get_slow_queries(
        self, 
        start_time: datetime, 
        end_time: datetime,
        threshold_ms: float = 1000.0
    ) -> List[Dict[str, Any]]:
        """Generate realistic slow query data."""
        slow_queries = []
        
        # Generate a few slow queries during the time period
        num_queries = random.randint(0, 5)
        
        for i in range(num_queries):
            query_time = start_time + timedelta(
                seconds=random.randint(0, int((end_time - start_time).total_seconds()))
            )
            
            queries = [
                "SELECT * FROM orders WHERE created_at > ? ORDER BY total DESC",
                "SELECT u.*, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id",
                "UPDATE inventory SET quantity = quantity - ? WHERE product_id = ?",
                "SELECT * FROM products p JOIN categories c ON p.category_id = c.id WHERE c.name = ?",
                "DELETE FROM sessions WHERE expires_at < NOW()"
            ]
            
            slow_query = {
                "query_id": f"slow_query_{i}_{int(query_time.timestamp())}",
                "timestamp": query_time,
                "query": random.choice(queries),
                "execution_time_ms": threshold_ms + random.uniform(0, 2000),
                "rows_examined": random.randint(1000, 50000),
                "rows_returned": random.randint(1, 1000),
                "table": random.choice(["orders", "users", "products", "inventory", "sessions"])
            }
            slow_queries.append(slow_query)
        
        return slow_queries
    
    async def get_lock_metrics(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[Metric]:
        """Generate realistic database lock metrics."""
        metrics = []
        interval = timedelta(minutes=1)
        
        current_time = start_time
        while current_time <= end_time:
            # Lock wait time
            base_lock_time = 5.0  # 5ms average lock wait
            
            value = base_lock_time + random.gauss(0, 2.0)
            
            # Simulate occasional lock contention
            if random.random() < 0.03:  # 3% chance
                value += random.uniform(50.0, 200.0)
            
            value = max(value, 0.0)
            
            metric = Metric(
                timestamp=current_time,
                value=value,
                metric_name="avg_lock_wait_time_ms",
                source="database_mock",
                metadata={"database": "main_db"}
            )
            metrics.append(metric)
            
            current_time += interval
        
        return metrics


class MockKnowledgeBaseClient(KnowledgeBaseClient):
    """Mock Knowledge Base client with predefined historical incidents."""
    
    def __init__(self, name: str = "mock_kb", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config or {})
        self._incidents = self._generate_historical_incidents()
    
    def _generate_historical_incidents(self) -> List[HistoricalIncident]:
        """Generate realistic historical incident data."""
        incidents = [
            HistoricalIncident(
                incident_id="INC-2024-001",
                timestamp=datetime.now() - timedelta(days=30),
                description="High latency on payment API due to database connection pool exhaustion",
                root_cause="Database connection pool reached maximum capacity during peak traffic",
                resolution="Increased connection pool size from 50 to 100 connections",
                metrics_pattern={
                    "response_time_ms": 2500.0,
                    "connection_pool_utilization_percent": 98.0,
                    "error_rate_percent": 15.0
                },
                resolution_time_minutes=45,
                tags=["database", "connection_pool", "payment_api", "high_traffic"]
            ),
            HistoricalIncident(
                incident_id="INC-2024-002",
                timestamp=datetime.now() - timedelta(days=25),
                description="Network latency spike causing API timeouts",
                root_cause="Network congestion due to DDoS attack on upstream provider",
                resolution="Activated DDoS protection and rerouted traffic through backup provider",
                metrics_pattern={
                    "response_time_ms": 5000.0,
                    "network_latency_ms": 150.0,
                    "packet_loss_percent": 5.0
                },
                resolution_time_minutes=120,
                tags=["network", "ddos", "latency", "upstream"]
            ),
            HistoricalIncident(
                incident_id="INC-2024-003",
                timestamp=datetime.now() - timedelta(days=20),
                description="Slow database queries causing application timeouts",
                root_cause="Missing database index on frequently queried column",
                resolution="Added composite index on orders(user_id, created_at)",
                metrics_pattern={
                    "response_time_ms": 3000.0,
                    "avg_query_time_ms": 800.0,
                    "avg_lock_wait_time_ms": 200.0
                },
                resolution_time_minutes=30,
                tags=["database", "slow_query", "index", "performance"]
            ),
            HistoricalIncident(
                incident_id="INC-2024-004",
                timestamp=datetime.now() - timedelta(days=15),
                description="Memory leak in application causing gradual performance degradation",
                root_cause="Unclosed database connections in background job processor",
                resolution="Fixed connection leak in job processor and restarted application",
                metrics_pattern={
                    "response_time_ms": 1800.0,
                    "connection_pool_utilization_percent": 85.0,
                    "error_rate_percent": 8.0
                },
                resolution_time_minutes=90,
                tags=["application", "memory_leak", "background_jobs", "connections"]
            ),
            HistoricalIncident(
                incident_id="INC-2024-005",
                timestamp=datetime.now() - timedelta(days=10),
                description="Third-party API rate limiting causing cascading failures",
                root_cause="Payment processor API rate limits exceeded during flash sale",
                resolution="Implemented exponential backoff and request queuing",
                metrics_pattern={
                    "response_time_ms": 4000.0,
                    "error_rate_percent": 25.0,
                    "requests_per_minute": 500.0
                },
                resolution_time_minutes=60,
                tags=["external", "rate_limiting", "payment", "flash_sale"]
            )
        ]
        return incidents
    
    async def connect(self) -> bool:
        """Simulate connection to knowledge base service."""
        await asyncio.sleep(0.1)
        self._update_connection_status(ConnectionStatus.CONNECTED)
        return True
    
    async def disconnect(self) -> None:
        """Simulate disconnection from knowledge base service."""
        await asyncio.sleep(0.05)
        self._update_connection_status(ConnectionStatus.DISCONNECTED)
    
    async def health_check(self) -> HealthCheckResult:
        """Perform mock health check."""
        start_time = datetime.now()
        await asyncio.sleep(0.02)
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        
        result = HealthCheckResult(
            status=ConnectionStatus.CONNECTED,
            response_time_ms=response_time
        )
        self._update_connection_status(ConnectionStatus.CONNECTED)
        self._last_health_check = result
        return result
    
    async def search_similar_incidents(
        self, 
        correlation_result: CorrelationResult,
        similarity_threshold: float = 0.7
    ) -> List[HistoricalIncident]:
        """Search for similar historical incidents based on correlation patterns."""
        similar_incidents = []
        
        spike_latency = correlation_result.spike_event.spike_latency
        
        for incident in self._incidents:
            # Simple similarity calculation based on response time and correlation patterns
            incident_latency = incident.metrics_pattern.get("response_time_ms", 0)
            
            # Calculate similarity score (simplified)
            latency_similarity = 1.0 - abs(spike_latency - incident_latency) / max(spike_latency, incident_latency, 1)
            
            # Check for matching correlation patterns
            pattern_matches = 0
            total_patterns = 0
            
            for metric, score in correlation_result.correlation_scores.items():
                if abs(score) > 0.5:  # Strong correlation
                    total_patterns += 1
                    # Check if incident has similar pattern
                    if metric.replace("_", " ") in incident.description.lower() or \
                       any(tag in metric for tag in incident.tags):
                        pattern_matches += 1
            
            pattern_similarity = pattern_matches / max(total_patterns, 1)
            
            # Combined similarity score
            overall_similarity = (latency_similarity * 0.6) + (pattern_similarity * 0.4)
            
            if overall_similarity >= similarity_threshold:
                similar_incidents.append(incident)
        
        # Sort by similarity (most similar first)
        return sorted(similar_incidents, key=lambda x: x.resolution_time_minutes)
    
    async def get_incident_by_id(self, incident_id: str) -> Optional[HistoricalIncident]:
        """Retrieve a specific incident by ID."""
        for incident in self._incidents:
            if incident.incident_id == incident_id:
                return incident
        return None
    
    async def store_incident(self, incident: HistoricalIncident) -> bool:
        """Store a new incident in the knowledge base."""
        # In a real implementation, this would persist to a database
        self._incidents.append(incident)
        return True
    
    async def get_common_patterns(self) -> List[Dict[str, Any]]:
        """Get common incident patterns and their typical causes."""
        patterns = [
            {
                "pattern_name": "Database Connection Pool Exhaustion",
                "indicators": ["high_connection_pool_utilization", "increased_response_time"],
                "typical_causes": ["Traffic spike", "Connection leak", "Insufficient pool size"],
                "common_resolutions": ["Increase pool size", "Fix connection leaks", "Add connection monitoring"],
                "frequency": 0.25
            },
            {
                "pattern_name": "Network Latency Spike",
                "indicators": ["high_network_latency", "packet_loss", "increased_response_time"],
                "typical_causes": ["Network congestion", "DDoS attack", "ISP issues"],
                "common_resolutions": ["DDoS protection", "Traffic rerouting", "Contact ISP"],
                "frequency": 0.20
            },
            {
                "pattern_name": "Slow Database Queries",
                "indicators": ["high_query_time", "lock_contention", "increased_response_time"],
                "typical_causes": ["Missing indexes", "Query optimization needed", "Lock contention"],
                "common_resolutions": ["Add indexes", "Optimize queries", "Review locking strategy"],
                "frequency": 0.30
            },
            {
                "pattern_name": "External Service Failure",
                "indicators": ["high_error_rate", "timeout_errors", "third_party_dependency"],
                "typical_causes": ["Third-party API issues", "Rate limiting", "Service outage"],
                "common_resolutions": ["Implement circuit breaker", "Add retry logic", "Use fallback service"],
                "frequency": 0.15
            },
            {
                "pattern_name": "Memory/Resource Exhaustion",
                "indicators": ["gradual_performance_degradation", "memory_usage_increase"],
                "typical_causes": ["Memory leak", "Resource not released", "Insufficient capacity"],
                "common_resolutions": ["Fix memory leaks", "Restart services", "Scale resources"],
                "frequency": 0.10
            }
        ]
        return patterns