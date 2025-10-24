# MCP client interfaces and implementations

from .base import (
    BaseMCPClient,
    APMClient,
    NetworkClient,
    DatabaseClient,
    KnowledgeBaseClient,
    HealthCheckResult,
    ConnectionStatus,
    HistoricalIncident
)

from .mock import (
    MockAPMClient,
    MockNetworkClient,
    MockDatabaseClient,
    MockKnowledgeBaseClient
)

from .manager import (
    MCPClientManager,
    ClientConfig,
    create_default_mock_manager
)

__all__ = [
    # Base classes
    'BaseMCPClient',
    'APMClient',
    'NetworkClient',
    'DatabaseClient',
    'KnowledgeBaseClient',
    'HealthCheckResult',
    'ConnectionStatus',
    'HistoricalIncident',
    
    # Mock implementations
    'MockAPMClient',
    'MockNetworkClient',
    'MockDatabaseClient',
    'MockKnowledgeBaseClient',
    
    # Manager
    'MCPClientManager',
    'ClientConfig',
    'create_default_mock_manager'
]