"""
Advanced Redis-based Distributed Caching Manager for COA/ROE System

This module provides comprehensive caching capabilities including:
- Redis-based distributed caching
- Cache invalidation strategies
- Cache warming mechanisms
- Performance monitoring
- Cache analytics
"""

import asyncio
import hashlib
import json
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import aioredis
import redis
from prometheus_client import Counter, Gauge, Histogram

import logging
# Use standard logging instead of FLAGSHIP.common
get_logger = logging.getLogger

logger = get_logger(__name__)

# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

CACHE_HITS = Counter(
    "coa_roe_cache_hits_total", "Total cache hits", ["tier", "operation"]
)
CACHE_MISSES = Counter(
    "coa_roe_cache_misses_total", "Total cache misses", ["tier", "operation"]
)
CACHE_OPERATIONS = Counter(
    "coa_roe_cache_operations_total", "Total cache operations", ["operation"]
)
CACHE_SIZE = Gauge("coa_roe_cache_size", "Cache size in bytes", ["tier"])
CACHE_LATENCY = Histogram(
    "coa_roe_cache_latency_seconds", "Cache operation latency", ["operation"]
)
CACHE_INVALIDATIONS = Counter(
    "coa_roe_cache_invalidations_total", "Total cache invalidations", ["reason"]
)

# ============================================================================
# CACHE TIERS
# ============================================================================


class CacheTier(str, Enum):
    """Cache tier levels."""

    L1 = "l1"  # In-memory (fastest)
    L2 = "l2"  # Redis (fast)
    L3 = "l3"  # Database (slowest)


# ============================================================================
# CACHE CONFIGURATION
# ============================================================================


@dataclass
class CacheConfig:
    """Cache configuration."""

    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    redis_ssl_cert_reqs: Optional[str] = None

    # TTL settings
    l1_ttl: int = 300  # 5 minutes
    l2_ttl: int = 3600  # 1 hour
    l3_ttl: int = 86400  # 24 hours

    # Cache warming settings
    enable_cache_warming: bool = True
    warming_batch_size: int = 100
    warming_interval: int = 300  # 5 minutes

    # Invalidation settings
    enable_auto_invalidation: bool = True
    invalidation_patterns: List[str] = None
    invalidation_interval: int = 60  # 1 minute

    # Performance settings
    max_memory_policy: str = "allkeys-lru"
    max_memory_size: str = "2gb"
    enable_compression: bool = True
    compression_threshold: int = 1024  # 1KB

    def __post_init__(self):
        if self.invalidation_patterns is None:
            self.invalidation_patterns = [
                "coa_roe:*",
                "rule_correlation:*",
                "compliance:*",
                "audit:*",
            ]


# ============================================================================
# CACHE MANAGER
# ============================================================================


class AdvancedCacheManager:
    """Advanced Redis-based distributed caching manager."""

    def __init__(self, config: CacheConfig):
        """Initialize the cache manager."""
        self.config = config
        self.logger = logger

        # Redis clients
        self.redis_client: Optional[redis.Redis] = None
        self.aioredis_client: Optional[aioredis.Redis] = None

        # In-memory cache (L1)
        self.l1_cache: Dict[str, Any] = {}
        self.l1_timestamps: Dict[str, float] = {}
        self.l1_lock = threading.RLock()

        # Cache statistics
        self.stats = {
            "l1_hits": 0,
            "l1_misses": 0,
            "l2_hits": 0,
            "l2_misses": 0,
            "l3_hits": 0,
            "l3_misses": 0,
            "invalidations": 0,
            "warm_ups": 0,
        }

        # Cache warming state
        self.warming_in_progress = False
        self.warming_lock = threading.Lock()

        # Invalidation state
        self.invalidation_in_progress = False
        self.invalidation_lock = threading.Lock()

        # Initialize Redis connection
        self._initialize_redis()

        # Start background tasks
        self._start_background_tasks()

        logger.info("Advanced Cache Manager initialized")

    def _initialize_redis(self):
        """Initialize Redis connection."""
        try:
            # Synchronous Redis client
            self.redis_client = redis.Redis.from_url(
                self.config.redis_url,
                db=self.config.redis_db,
                password=self.config.redis_password,
                ssl=self.config.redis_ssl,
                ssl_cert_reqs=self.config.redis_ssl_cert_reqs,
                decode_responses=True,
            )

            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established")

            # Configure Redis
            self._configure_redis()

        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self.redis_client = None

    async def _initialize_aioredis(self):
        """Initialize async Redis connection."""
        try:
            self.aioredis_client = aioredis.from_url(
                self.config.redis_url,
                db=self.config.redis_db,
                password=self.config.redis_password,
                ssl=self.config.redis_ssl,
                decode_responses=True,
            )

            # Test connection
            await self.aioredis_client.ping()
            logger.info("Async Redis connection established")

        except Exception as e:
            logger.error(f"Failed to initialize async Redis: {e}")
            self.aioredis_client = None

    def _configure_redis(self):
        """Configure Redis settings."""
        if not self.redis_client:
            return

        try:
            # Set memory policy
            self.redis_client.config_set(
                "maxmemory-policy", self.config.max_memory_policy
            )
            self.redis_client.config_set("maxmemory", self.config.max_memory_size)

            # Enable compression if configured
            if self.config.enable_compression:
                self.redis_client.config_set("save", "900 1 300 10 60 10000")

            logger.info("Redis configured successfully")

        except Exception as e:
            logger.warning(f"Failed to configure Redis: {e}")

    def _start_background_tasks(self):
        """Start background tasks for cache warming and invalidation."""
        if self.config.enable_cache_warming:
            threading.Thread(target=self._cache_warming_worker, daemon=True).start()

        if self.config.enable_auto_invalidation:
            threading.Thread(
                target=self._cache_invalidation_worker, daemon=True
            ).start()

    def _generate_key(
        self, key: str, tier: CacheTier, namespace: str = "coa_roe"
    ) -> str:
        """Generate cache key with tier prefix."""
        return f"{namespace}:{tier.value}:{hashlib.md5(key.encode()).hexdigest()}"

    def _is_expired(self, timestamp: float, ttl: int) -> bool:
        """Check if cache entry is expired."""
        return time.time() - timestamp > ttl

    def _compress_data(self, data: Any) -> bytes:
        """Compress data if enabled and above threshold."""
        if not self.config.enable_compression:
            return json.dumps(data).encode()

        import gzip

        json_data = json.dumps(data)

        if len(json_data) < self.config.compression_threshold:
            return json_data.encode()

        return gzip.compress(json_data.encode())

    def _decompress_data(self, data: bytes) -> Any:
        """Decompress data if compressed."""
        if not self.config.enable_compression:
            return json.loads(data.decode())

        try:

            decompressed = gzip.decompress(data)
            return json.loads(decompressed.decode())
        except:
            # Fallback to regular JSON
            return json.loads(data.decode())

    async def get(
        self, key: str, default: Any = None, namespace: str = "coa_roe"
    ) -> Any:
        """Get value from multi-tier cache."""
        start_time = time.time()

        # Try L1 cache first
        with self.l1_lock:
            if key in self.l1_cache:
                if not self._is_expired(self.l1_timestamps[key], self.config.l1_ttl):
                    self.stats["l1_hits"] += 1
                    CACHE_HITS.labels(tier="l1", operation="get").inc()
                    CACHE_LATENCY.labels(operation="get").observe(
                        time.time() - start_time
                    )
                    return self.l1_cache[key]
                else:
                    # Remove expired entry
                    del self.l1_cache[key]
                    del self.l1_timestamps[key]

        self.stats["l1_misses"] += 1
        CACHE_MISSES.labels(tier="l1", operation="get").inc()

        # Try L2 cache (Redis)
        if self.redis_client:
            try:
                l2_key = self._generate_key(key, CacheTier.L2, namespace)
                value = self.redis_client.get(l2_key)

                if value:
                    # Decompress and cache in L1
                    data = self._decompress_data(value.encode())
                    with self.l1_lock:
                        self.l1_cache[key] = data
                        self.l1_timestamps[key] = time.time()

                    self.stats["l2_hits"] += 1
                    CACHE_HITS.labels(tier="l2", operation="get").inc()
                    CACHE_LATENCY.labels(operation="get").observe(
                        time.time() - start_time
                    )
                    return data

            except Exception as e:
                logger.warning(f"L2 cache error: {e}")

        self.stats["l2_misses"] += 1
        CACHE_MISSES.labels(tier="l2", operation="get").inc()

        # Try L3 cache (Database) - implemented separately
        # For now, return default
        CACHE_LATENCY.labels(operation="get").observe(time.time() - start_time)
        return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        namespace: str = "coa_roe",
        tier: CacheTier = CacheTier.L2,
    ) -> bool:
        """Set value in cache."""
        start_time = time.time()

        try:
            # Set in L1
            with self.l1_lock:
                self.l1_cache[key] = value
                self.l1_timestamps[key] = time.time()

            # Set in L2 (Redis)
            if self.redis_client and tier in [CacheTier.L2, CacheTier.L3]:
                l2_key = self._generate_key(key, CacheTier.L2, namespace)
                l2_ttl = ttl or self.config.l2_ttl
                compressed_data = self._compress_data(value)
                self.redis_client.setex(l2_key, l2_ttl, compressed_data)

            # Set in L3 (Database) - only if TTL is reasonable
            if tier == CacheTier.L3:
                effective_ttl = ttl or self.config.l3_ttl
                if effective_ttl > 1:  # Only store in L3 if TTL > 1 second
                    # Implement L3 storage here
                    pass

            CACHE_OPERATIONS.labels(operation="set").inc()
            CACHE_LATENCY.labels(operation="set").observe(time.time() - start_time)
            return True

        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    async def delete(self, key: str, namespace: str = "coa_roe") -> bool:
        """Delete value from all cache tiers."""
        start_time = time.time()

        try:
            # Delete from L1
            with self.l1_lock:
                if key in self.l1_cache:
                    del self.l1_cache[key]
                    del self.l1_timestamps[key]

            # Delete from L2
            if self.redis_client:
                l2_key = self._generate_key(key, CacheTier.L2, namespace)
                self.redis_client.delete(l2_key)

            # Delete from L3
            # Implement L3 deletion here

            CACHE_OPERATIONS.labels(operation="delete").inc()
            CACHE_LATENCY.labels(operation="delete").observe(time.time() - start_time)
            return True

        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    async def invalidate_pattern(self, pattern: str, namespace: str = "coa_roe") -> int:
        """Invalidate cache entries matching pattern."""
        if not self.redis_client:
            return 0

        try:
            # Invalidate L1 cache entries matching pattern
            with self.l1_lock:
                keys_to_delete = [k for k in self.l1_cache.keys() if pattern in k]
                for key in keys_to_delete:
                    del self.l1_cache[key]
                    del self.l1_timestamps[key]

            # Invalidate L2 cache entries matching pattern
            l2_pattern = f"{namespace}:l2:*{pattern}*"
            keys = self.redis_client.keys(l2_pattern)
            if keys:
                self.redis_client.delete(*keys)

            CACHE_INVALIDATIONS.labels(reason="pattern").inc()
            self.stats["invalidations"] += len(keys_to_delete) + len(keys)

            return len(keys_to_delete) + len(keys)

        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
            return 0

    async def warm_cache(self, keys: List[str], namespace: str = "coa_roe") -> int:
        """Warm cache with specified keys."""
        if not self.redis_client:
            return 0

        warmed_count = 0

        for key in keys:
            try:
                # Check if key exists in L2
                l2_key = self._generate_key(key, CacheTier.L2, namespace)
                value = self.redis_client.get(l2_key)

                if value:
                    # Load into L1 cache
                    data = self._decompress_data(value.encode())
                    with self.l1_lock:
                        self.l1_cache[key] = data
                        self.l1_timestamps[key] = time.time()
                    warmed_count += 1

            except Exception as e:
                logger.warning(f"Cache warming error for key {key}: {e}")

        self.stats["warm_ups"] += warmed_count
        return warmed_count

    def _cache_warming_worker(self):
        """Background worker for cache warming."""
        while True:
            try:
                if self.warming_in_progress:
                    time.sleep(1)
                    continue

                with self.warming_lock:
                    self.warming_in_progress = True

                # Get frequently accessed keys from Redis
                if self.redis_client:
                    # Get keys with highest access frequency
                    keys = self.redis_client.keys("coa_roe:l2:*")

                    if keys:
                        # Warm cache with most frequently accessed keys
                        keys_to_warm = keys[: self.config.warming_batch_size]
                        asyncio.run(self.warm_cache(keys_to_warm))

                with self.warming_lock:
                    self.warming_in_progress = False

                time.sleep(self.config.warming_interval)

            except Exception as e:
                logger.error(f"Cache warming worker error: {e}")
                with self.warming_lock:
                    self.warming_in_progress = False
                time.sleep(60)

    def _cache_invalidation_worker(self):
        """Background worker for cache invalidation."""
        while True:
            try:
                if self.invalidation_in_progress:
                    time.sleep(1)
                    continue

                with self.invalidation_lock:
                    self.invalidation_in_progress = True

                # Invalidate expired entries
                current_time = time.time()

                with self.l1_lock:
                    expired_keys = [
                        key
                        for key, timestamp in self.l1_timestamps.items()
                        if self._is_expired(timestamp, self.config.l1_ttl)
                    ]

                    for key in expired_keys:
                        del self.l1_cache[key]
                        del self.l1_timestamps[key]

                if expired_keys:
                    CACHE_INVALIDATIONS.labels(reason="expired").inc()
                    self.stats["invalidations"] += len(expired_keys)

                with self.invalidation_lock:
                    self.invalidation_in_progress = False

                time.sleep(self.config.invalidation_interval)

            except Exception as e:
                logger.error(f"Cache invalidation worker error: {e}")
                with self.invalidation_lock:
                    self.invalidation_in_progress = False
                time.sleep(60)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = (
            self.stats["l1_hits"] + self.stats["l2_hits"] + self.stats["l3_hits"]
        )
        total_misses = (
            self.stats["l1_misses"] + self.stats["l2_misses"] + self.stats["l3_misses"]
        )
        total_requests = total_hits + total_misses

        hit_ratio = total_hits / total_requests if total_requests > 0 else 0

        return {
            "stats": self.stats,
            "hit_ratio": hit_ratio,
            "total_requests": total_requests,
            "l1_size": len(self.l1_cache),
            "l2_size": (
                len(self.redis_client.keys("coa_roe:l2:*")) if self.redis_client else 0
            ),
            "warming_in_progress": self.warming_in_progress,
            "invalidation_in_progress": self.invalidation_in_progress,
        }

    async def clear_all(self, namespace: str = "coa_roe") -> bool:
        """Clear all cache entries."""
        try:
            # Clear L1 cache
            with self.l1_lock:
                self.l1_cache.clear()
                self.l1_timestamps.clear()

            # Clear L2 cache
            if self.redis_client:
                keys = self.redis_client.keys(f"{namespace}:*")
                if keys:
                    self.redis_client.delete(*keys)

            # Clear L3 cache
            # Implement L3 clearing here

            logger.info("All cache entries cleared")
            return True

        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
