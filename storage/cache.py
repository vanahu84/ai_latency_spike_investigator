"""
Redis caching layer for real-time metrics and correlation results.

This module provides caching functionality for frequently accessed data
to improve performance and reduce load on external APIs.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import redis
from redis.exceptions import RedisError, ConnectionError

from config import config
from models.core import Metric, CorrelationResult, SpikeEvent


logger = logging.getLogger(__name__)


class CacheError(Exception):
    """Custom exception for cache operations."""
    pass


class CacheManager:
    """Manages Redis caching operations with proper error handling."""
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize cache manager with optional custom Redis URL."""
        self.redis_url = redis_url or config.REDIS_URL
        self._client = None
        self._connected = False
        self.default_ttl = config.CACHE_TTL_SECONDS
        self._connection_retries = 3
        self._retry_delay = 1  # seconds
    
    def _get_client(self) -> redis.Redis:
        """Get Redis client with connection management and retry logic."""
        if self._client is None or not self._connected:
            for attempt in range(self._connection_retries):
                try:
                    self._client = redis.from_url(
                        self.redis_url,
                        decode_responses=True,
                        socket_timeout=5,
                        socket_connect_timeout=5,
                        retry_on_timeout=True,
                        health_check_interval=30
                    )
                    # Test connection
                    self._client.ping()
                    self._connected = True
                    logger.info("Redis connection established")
                    break
                except (RedisError, ConnectionError) as e:
                    if attempt < self._connection_retries - 1:
                        logger.warning(f"Redis connection attempt {attempt + 1} failed: {e}, retrying...")
                        import time
                        time.sleep(self._retry_delay * (attempt + 1))  # Exponential backoff
                    else:
                        logger.warning(f"Redis connection failed after {self._connection_retries} attempts: {e}")
                        self._connected = False
                        raise CacheError(f"Failed to connect to Redis: {e}")
        
        return self._client
    
    def is_available(self) -> bool:
        """Check if Redis cache is available."""
        try:
            client = self._get_client()
            client.ping()
            return True
        except Exception:
            return False
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache with optional TTL."""
        if not self.is_available():
            logger.warning("Cache not available, skipping set operation")
            return False
        
        try:
            client = self._get_client()
            ttl = ttl or self.default_ttl
            
            # Serialize complex objects to JSON
            if isinstance(value, (dict, list)):
                value = json.dumps(value, default=self._json_serializer)
            elif hasattr(value, 'to_dict'):
                value = json.dumps(value.to_dict())
            
            result = client.setex(key, ttl, value)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from cache."""
        if not self.is_available():
            logger.warning("Cache not available, returning default")
            return default
        
        try:
            client = self._get_client()
            value = client.get(key)
            
            if value is None:
                return default
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return default
    
    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        if not self.is_available():
            return False
        
        try:
            client = self._get_client()
            result = client.delete(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        if not self.is_available():
            return False
        
        try:
            client = self._get_client()
            return bool(client.exists(key))
        except Exception as e:
            logger.error(f"Failed to check cache key {key}: {e}")
            return False
    
    def set_metrics(self, metrics: List[Metric], ttl: Optional[int] = None) -> bool:
        """Cache a list of metrics with timestamp-based keys."""
        if not metrics:
            return True
        
        try:
            # Group metrics by source and metric name
            grouped_metrics = {}
            for metric in metrics:
                key = f"metrics:{metric.source}:{metric.metric_name}"
                if key not in grouped_metrics:
                    grouped_metrics[key] = []
                grouped_metrics[key].append({
                    'timestamp': metric.timestamp.isoformat(),
                    'value': metric.value,
                    'metadata': metric.metadata
                })
            
            # Store each group
            success = True
            for key, metric_list in grouped_metrics.items():
                if not self.set(key, metric_list, ttl or 3600):  # 1 hour default for metrics
                    success = False
            
            return success
        except Exception as e:
            logger.error(f"Failed to cache metrics: {e}")
            return False
    
    def get_metrics(self, source: str, metric_name: str, 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get cached metrics for a specific source and metric name."""
        key = f"metrics:{source}:{metric_name}"
        cached_data = self.get(key, [])
        
        if not cached_data or not isinstance(cached_data, list):
            return []
        
        # Filter by time range if specified
        if start_time or end_time:
            filtered_data = []
            for metric_data in cached_data:
                try:
                    metric_time = datetime.fromisoformat(metric_data['timestamp'])
                    if start_time and metric_time < start_time:
                        continue
                    if end_time and metric_time > end_time:
                        continue
                    filtered_data.append(metric_data)
                except (KeyError, ValueError):
                    continue
            return filtered_data
        
        return cached_data
    
    def cache_correlation_result(self, correlation: CorrelationResult, ttl: Optional[int] = None) -> bool:
        """Cache a correlation result."""
        key = f"correlation:{correlation.spike_event.endpoint}:{correlation.spike_event.timestamp.isoformat()}"
        return self.set(key, correlation.to_dict(), ttl or 86400)  # 24 hours default
    
    def get_cached_correlation(self, endpoint: str, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Get cached correlation result."""
        key = f"correlation:{endpoint}:{timestamp.isoformat()}"
        return self.get(key)
    
    def cache_spike_event(self, spike_event: SpikeEvent, ttl: Optional[int] = None) -> bool:
        """Cache a spike event."""
        key = f"spike:{spike_event.endpoint}:{spike_event.timestamp.isoformat()}"
        return self.set(key, spike_event.to_dict(), ttl or 86400)  # 24 hours default
    
    def get_recent_spikes(self, endpoint: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recently cached spike events."""
        try:
            client = self._get_client()
            pattern = f"spike:{endpoint}:*" if endpoint else "spike:*"
            keys = client.keys(pattern)
            
            # Sort keys by timestamp (newest first) and limit
            sorted_keys = sorted(keys, reverse=True)[:limit]
            
            spikes = []
            for key in sorted_keys:
                spike_data = self.get(key)
                if spike_data:
                    spikes.append(spike_data)
            
            return spikes
        except Exception as e:
            logger.error(f"Failed to get recent spikes: {e}")
            return []
    
    def set_api_response_cache(self, api_name: str, endpoint: str, params: Dict[str, Any], 
                              response: Any, ttl: Optional[int] = None) -> bool:
        """Cache API response to reduce external calls."""
        # Create a cache key from API name, endpoint, and parameters
        params_str = json.dumps(params, sort_keys=True)
        key = f"api:{api_name}:{endpoint}:{hash(params_str)}"
        return self.set(key, response, ttl or 300)  # 5 minutes default for API responses
    
    def get_api_response_cache(self, api_name: str, endpoint: str, params: Dict[str, Any]) -> Any:
        """Get cached API response."""
        params_str = json.dumps(params, sort_keys=True)
        key = f"api:{api_name}:{endpoint}:{hash(params_str)}"
        return self.get(key)
    
    def clear_expired_keys(self) -> int:
        """Clear expired keys (Redis handles this automatically, but useful for manual cleanup)."""
        if not self.is_available():
            return 0
        
        try:
            client = self._get_client()
            # Get all keys with TTL
            keys_with_ttl = []
            for key in client.scan_iter():
                ttl = client.ttl(key)
                if ttl == -2:  # Key doesn't exist
                    keys_with_ttl.append(key)
            
            if keys_with_ttl:
                return client.delete(*keys_with_ttl)
            return 0
        except Exception as e:
            logger.error(f"Failed to clear expired keys: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        if not self.is_available():
            return {'available': False}
        
        try:
            client = self._get_client()
            info = client.info()
            
            # Get key counts by pattern
            key_counts = {}
            try:
                key_counts['metrics'] = len(client.keys('metrics:*'))
                key_counts['correlations'] = len(client.keys('correlation:*'))
                key_counts['spikes'] = len(client.keys('spike:*'))
                key_counts['api_responses'] = len(client.keys('api:*'))
                key_counts['total'] = sum(key_counts.values())
            except Exception as e:
                logger.warning(f"Failed to get key counts: {e}")
                key_counts = {'error': 'Unable to count keys'}
            
            return {
                'available': True,
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory_human', '0B'),
                'used_memory_bytes': info.get('used_memory', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': self._calculate_hit_rate(info),
                'key_counts': key_counts,
                'uptime_seconds': info.get('uptime_in_seconds', 0),
                'redis_version': info.get('redis_version', 'unknown')
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {'available': False, 'error': str(e)}
    
    def _calculate_hit_rate(self, info: Dict[str, Any]) -> float:
        """Calculate cache hit rate."""
        hits = info.get('keyspace_hits', 0)
        misses = info.get('keyspace_misses', 0)
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0
    
    def _json_serializer(self, obj: Any) -> str:
        """Custom JSON serializer for datetime and other objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, timedelta):
            return obj.total_seconds()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def health_check(self) -> bool:
        """Check if cache is healthy."""
        try:
            client = self._get_client()
            client.ping()
            return True
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return False
    
    def flush_all(self) -> bool:
        """Flush all cache data (use with caution)."""
        if not self.is_available():
            return False
        
        try:
            client = self._get_client()
            client.flushall()
            logger.warning("All cache data has been flushed")
            return True
        except Exception as e:
            logger.error(f"Failed to flush cache: {e}")
            return False