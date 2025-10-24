"""
MCP Client Manager for connection management and health monitoring.

This module provides centralized management of all MCP clients with health checking,
connection pooling, and error handling capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass

from .base import (
    BaseMCPClient, APMClient, NetworkClient, DatabaseClient, KnowledgeBaseClient,
    HealthCheckResult, ConnectionStatus
)
from .mock import MockAPMClient, MockNetworkClient, MockDatabaseClient, MockKnowledgeBaseClient


logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Configuration for a single MCP client."""
    client_type: str  # 'apm', 'network', 'database', 'knowledge_base'
    implementation: str  # 'mock', 'newrelic', 'datadog', etc.
    config: Dict[str, Any]
    enabled: bool = True
    health_check_interval_seconds: int = 60


class MCPClientManager:
    """Manages all MCP clients with health monitoring and connection management."""
    
    def __init__(self, configs: List[ClientConfig]):
        self.configs = configs
        self.clients: Dict[str, BaseMCPClient] = {}
        self._health_check_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        
        # Client type mappings
        self._client_classes = {
            'apm': {
                'mock': MockAPMClient,
                # Add real implementations here
                # 'newrelic': NewRelicAPMClient,
                # 'datadog': DatadogAPMClient,
            },
            'network': {
                'mock': MockNetworkClient,
                # Add real implementations here
                # 'prometheus': PrometheusNetworkClient,
            },
            'database': {
                'mock': MockDatabaseClient,
                # Add real implementations here
                # 'postgresql': PostgreSQLClient,
            },
            'knowledge_base': {
                'mock': MockKnowledgeBaseClient,
                # Add real implementations here
                # 'elasticsearch': ElasticsearchKBClient,
            }
        }
    
    async def initialize(self) -> bool:
        """Initialize all configured clients."""
        logger.info("Initializing MCP clients...")
        
        success_count = 0
        for config in self.configs:
            if not config.enabled:
                logger.info(f"Skipping disabled client: {config.client_type}:{config.implementation}")
                continue
            
            try:
                client = await self._create_client(config)
                if client:
                    client_key = f"{config.client_type}:{config.implementation}"
                    self.clients[client_key] = client
                    success_count += 1
                    logger.info(f"Successfully initialized client: {client_key}")
                else:
                    logger.error(f"Failed to create client: {config.client_type}:{config.implementation}")
            except Exception as e:
                logger.error(f"Error initializing client {config.client_type}:{config.implementation}: {e}")
        
        logger.info(f"Initialized {success_count}/{len([c for c in self.configs if c.enabled])} clients")
        return success_count > 0
    
    async def _create_client(self, config: ClientConfig) -> Optional[BaseMCPClient]:
        """Create a client instance based on configuration."""
        client_classes = self._client_classes.get(config.client_type, {})
        client_class = client_classes.get(config.implementation)
        
        if not client_class:
            logger.error(f"Unknown client implementation: {config.client_type}:{config.implementation}")
            return None
        
        try:
            client = client_class(
                name=f"{config.client_type}_{config.implementation}",
                config=config.config
            )
            
            # Attempt to connect
            connected = await client.connect()
            if not connected:
                logger.warning(f"Failed to connect client: {config.client_type}:{config.implementation}")
                return None
            
            return client
        except Exception as e:
            logger.error(f"Error creating client {config.client_type}:{config.implementation}: {e}")
            return None
    
    async def start_health_monitoring(self) -> None:
        """Start health check monitoring for all clients."""
        if self._running:
            logger.warning("Health monitoring is already running")
            return
        
        self._running = True
        logger.info("Starting health check monitoring...")
        
        for client_key, client in self.clients.items():
            config = self._get_config_for_client(client_key)
            if config:
                task = asyncio.create_task(
                    self._health_check_loop(client, config.health_check_interval_seconds)
                )
                self._health_check_tasks[client_key] = task
        
        logger.info(f"Started health monitoring for {len(self._health_check_tasks)} clients")
    
    async def stop_health_monitoring(self) -> None:
        """Stop health check monitoring for all clients."""
        if not self._running:
            return
        
        self._running = False
        logger.info("Stopping health check monitoring...")
        
        # Cancel all health check tasks
        for task in self._health_check_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self._health_check_tasks:
            await asyncio.gather(*self._health_check_tasks.values(), return_exceptions=True)
        
        self._health_check_tasks.clear()
        logger.info("Health check monitoring stopped")
    
    async def _health_check_loop(self, client: BaseMCPClient, interval_seconds: int) -> None:
        """Continuous health check loop for a single client."""
        while self._running:
            try:
                await client.health_check()
                logger.debug(f"Health check completed for {client.name}")
            except Exception as e:
                logger.error(f"Health check failed for {client.name}: {e}")
            
            try:
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
    
    def _get_config_for_client(self, client_key: str) -> Optional[ClientConfig]:
        """Get configuration for a specific client key."""
        client_type, implementation = client_key.split(':', 1)
        for config in self.configs:
            if config.client_type == client_type and config.implementation == implementation:
                return config
        return None
    
    def get_apm_client(self) -> Optional[APMClient]:
        """Get the first available APM client."""
        for client_key, client in self.clients.items():
            if client_key.startswith('apm:') and isinstance(client, APMClient):
                if client.is_connected:
                    return client
        return None
    
    def get_network_client(self) -> Optional[NetworkClient]:
        """Get the first available Network client."""
        for client_key, client in self.clients.items():
            if client_key.startswith('network:') and isinstance(client, NetworkClient):
                if client.is_connected:
                    return client
        return None
    
    def get_database_client(self) -> Optional[DatabaseClient]:
        """Get the first available Database client."""
        for client_key, client in self.clients.items():
            if client_key.startswith('database:') and isinstance(client, DatabaseClient):
                if client.is_connected:
                    return client
        return None
    
    def get_knowledge_base_client(self) -> Optional[KnowledgeBaseClient]:
        """Get the first available Knowledge Base client."""
        for client_key, client in self.clients.items():
            if client_key.startswith('knowledge_base:') and isinstance(client, KnowledgeBaseClient):
                if client.is_connected:
                    return client
        return None
    
    def get_all_clients(self) -> Dict[str, BaseMCPClient]:
        """Get all registered clients."""
        return self.clients.copy()
    
    async def health_check_all(self) -> Dict[str, HealthCheckResult]:
        """Perform health check on all clients and return results."""
        results = {}
        
        tasks = []
        client_keys = []
        
        for client_key, client in self.clients.items():
            tasks.append(client.health_check())
            client_keys.append(client_key)
        
        if tasks:
            health_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(health_results):
                client_key = client_keys[i]
                if isinstance(result, Exception):
                    results[client_key] = HealthCheckResult(
                        status=ConnectionStatus.ERROR,
                        response_time_ms=0.0,
                        error_message=str(result)
                    )
                else:
                    results[client_key] = result
        
        return results
    
    def get_client_status_summary(self) -> Dict[str, Any]:
        """Get a summary of all client statuses."""
        summary = {
            'total_clients': len(self.clients),
            'connected_clients': 0,
            'disconnected_clients': 0,
            'error_clients': 0,
            'clients': {}
        }
        
        for client_key, client in self.clients.items():
            status = client._connection_status
            last_check = client.last_health_check
            
            client_info = {
                'status': status.value,
                'last_health_check': last_check.last_check.isoformat() if last_check else None,
                'last_response_time_ms': last_check.response_time_ms if last_check else None,
                'last_error': last_check.error_message if last_check else None
            }
            
            summary['clients'][client_key] = client_info
            
            if status == ConnectionStatus.CONNECTED:
                summary['connected_clients'] += 1
            elif status == ConnectionStatus.DISCONNECTED:
                summary['disconnected_clients'] += 1
            else:
                summary['error_clients'] += 1
        
        return summary
    
    async def shutdown(self) -> None:
        """Shutdown all clients and cleanup resources."""
        logger.info("Shutting down MCP Client Manager...")
        
        # Stop health monitoring
        await self.stop_health_monitoring()
        
        # Disconnect all clients
        disconnect_tasks = []
        for client in self.clients.values():
            disconnect_tasks.append(client.disconnect())
        
        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        
        self.clients.clear()
        logger.info("MCP Client Manager shutdown complete")


def create_default_mock_manager() -> MCPClientManager:
    """Create a client manager with default mock configurations for development."""
    configs = [
        ClientConfig(
            client_type='apm',
            implementation='mock',
            config={},
            enabled=True,
            health_check_interval_seconds=30
        ),
        ClientConfig(
            client_type='network',
            implementation='mock',
            config={},
            enabled=True,
            health_check_interval_seconds=30
        ),
        ClientConfig(
            client_type='database',
            implementation='mock',
            config={},
            enabled=True,
            health_check_interval_seconds=30
        ),
        ClientConfig(
            client_type='knowledge_base',
            implementation='mock',
            config={},
            enabled=True,
            health_check_interval_seconds=60
        )
    ]
    
    return MCPClientManager(configs)