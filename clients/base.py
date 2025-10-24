"""
Abstract base classes for MCP client interfaces.

This module defines the contract for all MCP clients used in the Latency Spike
Root Cause Investigator system.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from models import Metric, SpikeEvent, CorrelationResult


class ConnectionStatus(Enum):
    """Enumeration for client connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a client health check."""
    status: ConnectionStatus
    response_time_ms: float
    error_message: Optional[str] = None
    last_check: datetime = None
    
    def __post_init__(self):
        if self.last_check is None:
            self.last_check = datetime.now()


class BaseMCPClient(ABC):
    """Abstract base class for all MCP clients."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self._connection_status = ConnectionStatus.UNKNOWN
        self._last_health_check: Optional[HealthCheckResult] = None
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the MCP server."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the MCP server."""
        pass
    
    @abstractmethod
    async def health_check(self) -> HealthCheckResult:
        """Perform health check and return status."""
        pass
    
    @property
    def is_connected(self) -> bool:
        """Check if client is currently connected."""
        return self._connection_status == ConnectionStatus.CONNECTED
    
    @property
    def last_health_check(self) -> Optional[HealthCheckResult]:
        """Get the last health check result."""
        return self._last_health_check
    
    def _update_connection_status(self, status: ConnectionStatus) -> None:
        """Update the internal connection status."""
        self._connection_status = status


class APMClient(BaseMCPClient):
    """Abstract base class for APM (Application Performance Monitoring) clients."""
    
    @abstractmethod
    async def get_latency_metrics(
        self, 
        start_time: datetime, 
        end_time: datetime,
        endpoint: Optional[str] = None
    ) -> List[Metric]:
        """Retrieve latency metrics for the specified time range."""
        pass
    
    @abstractmethod
    async def get_throughput_metrics(
        self, 
        start_time: datetime, 
        end_time: datetime,
        endpoint: Optional[str] = None
    ) -> List[Metric]:
        """Retrieve throughput metrics for the specified time range."""
        pass
    
    @abstractmethod
    async def get_error_rate_metrics(
        self, 
        start_time: datetime, 
        end_time: datetime,
        endpoint: Optional[str] = None
    ) -> List[Metric]:
        """Retrieve error rate metrics for the specified time range."""
        pass
    
    @abstractmethod
    async def get_available_endpoints(self) -> List[str]:
        """Get list of available API endpoints being monitored."""
        pass


class NetworkClient(BaseMCPClient):
    """Abstract base class for Network monitoring clients."""
    
    @abstractmethod
    async def get_bandwidth_metrics(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[Metric]:
        """Retrieve network bandwidth metrics."""
        pass
    
    @abstractmethod
    async def get_latency_metrics(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[Metric]:
        """Retrieve network latency metrics."""
        pass
    
    @abstractmethod
    async def get_packet_loss_metrics(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[Metric]:
        """Retrieve packet loss metrics."""
        pass
    
    @abstractmethod
    async def get_connection_count_metrics(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[Metric]:
        """Retrieve active connection count metrics."""
        pass


class DatabaseClient(BaseMCPClient):
    """Abstract base class for Database monitoring clients."""
    
    @abstractmethod
    async def get_query_performance_metrics(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[Metric]:
        """Retrieve database query performance metrics."""
        pass
    
    @abstractmethod
    async def get_connection_pool_metrics(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[Metric]:
        """Retrieve database connection pool metrics."""
        pass
    
    @abstractmethod
    async def get_slow_queries(
        self, 
        start_time: datetime, 
        end_time: datetime,
        threshold_ms: float = 1000.0
    ) -> List[Dict[str, Any]]:
        """Retrieve slow queries above the threshold."""
        pass
    
    @abstractmethod
    async def get_lock_metrics(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[Metric]:
        """Retrieve database lock contention metrics."""
        pass


@dataclass
class HistoricalIncident:
    """Represents a historical incident for pattern matching."""
    incident_id: str
    timestamp: datetime
    description: str
    root_cause: str
    resolution: str
    metrics_pattern: Dict[str, float]
    resolution_time_minutes: int
    tags: List[str]


class KnowledgeBaseClient(BaseMCPClient):
    """Abstract base class for Knowledge Base clients."""
    
    @abstractmethod
    async def search_similar_incidents(
        self, 
        correlation_result: CorrelationResult,
        similarity_threshold: float = 0.7
    ) -> List[HistoricalIncident]:
        """Search for similar historical incidents."""
        pass
    
    @abstractmethod
    async def get_incident_by_id(self, incident_id: str) -> Optional[HistoricalIncident]:
        """Retrieve a specific incident by ID."""
        pass
    
    @abstractmethod
    async def store_incident(self, incident: HistoricalIncident) -> bool:
        """Store a new incident in the knowledge base."""
        pass
    
    @abstractmethod
    async def get_common_patterns(self) -> List[Dict[str, Any]]:
        """Get common incident patterns and their typical causes."""
        pass