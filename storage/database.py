"""
SQLite database management for persistent storage.

This module handles database connections, table creation, and basic operations
for storing spike events, correlations, and recommendations.
"""

import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator
import json

from config import config
from models.core import SpikeEvent, CorrelationResult, RootCause, SeverityLevel


logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Custom exception for database operations."""
    pass


class DatabaseManager:
    """Manages SQLite database operations with proper error handling."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database manager with optional custom path."""
        self.db_path = db_path or config.SQLITE_DB_PATH
        self._ensure_db_directory()
        self._initialized = False
        self._schema_version = 1  # Current schema version
    
    def _ensure_db_directory(self) -> None:
        """Ensure the database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections with proper cleanup."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
            conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
            yield conn
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            if conn:
                conn.close()
    
    def initialize_database(self) -> None:
        """Initialize database with required tables."""
        if self._initialized:
            return
        
        try:
            # Check if migration is needed
            if self.needs_migration():
                self.migrate_schema()
            else:
                # Ensure tables exist (for new databases)
                with self.get_connection() as conn:
                    self._create_tables(conn)
                    conn.commit()
            
            self._initialized = True
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")
    
    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """Create all required database tables."""
        
        # Schema version table for migrations
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Spike events table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS spike_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                severity TEXT NOT NULL,
                baseline_latency REAL NOT NULL,
                spike_latency REAL NOT NULL,
                duration_seconds REAL NOT NULL,
                affected_metrics TEXT,  -- JSON array
                metadata TEXT,  -- JSON object
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Correlations table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS correlations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                spike_event_id INTEGER NOT NULL,
                network_metrics TEXT,  -- JSON object
                db_metrics TEXT,  -- JSON object
                correlation_scores TEXT,  -- JSON object
                confidence REAL NOT NULL,
                analysis_timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (spike_event_id) REFERENCES spike_events (id) ON DELETE CASCADE
            )
        """)
        
        # Recommendations table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                correlation_id INTEGER NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                supporting_evidence TEXT,  -- JSON array
                recommended_actions TEXT,  -- JSON array
                priority INTEGER NOT NULL DEFAULT 1,
                status TEXT DEFAULT 'pending',  -- pending, completed, dismissed
                feedback TEXT,  -- User feedback
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (correlation_id) REFERENCES correlations (id) ON DELETE CASCADE
            )
        """)
        
        # Recommendation feedback table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS recommendation_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recommendation_id TEXT NOT NULL,
                incident_id TEXT,
                effectiveness TEXT NOT NULL,  -- Very Helpful, Somewhat Helpful, Not Helpful, Made it Worse
                implementation TEXT NOT NULL,  -- Successfully Implemented, Partially Implemented, Could Not Implement, Not Attempted
                time_to_implement TEXT,  -- < 5 minutes, 5-15 minutes, etc.
                feedback_text TEXT,
                timestamp TEXT NOT NULL,
                user_id TEXT DEFAULT 'anonymous',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Action completions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS action_completions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id TEXT NOT NULL,
                action_key TEXT NOT NULL,
                action_text TEXT NOT NULL,
                completed BOOLEAN NOT NULL DEFAULT 0,
                timestamp TEXT NOT NULL,
                user_id TEXT DEFAULT 'anonymous',
                completion_notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(incident_id, action_key)
            )
        """)
        
        # Create indexes for better query performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_spike_events_timestamp ON spike_events(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_spike_events_endpoint ON spike_events(endpoint)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_spike_events_severity ON spike_events(severity)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_correlations_analysis_timestamp ON correlations(analysis_timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_correlations_spike_event_id ON correlations(spike_event_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_priority ON recommendations(priority)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_status ON recommendations(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_recommendation_feedback_incident ON recommendation_feedback(incident_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_recommendation_feedback_timestamp ON recommendation_feedback(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_action_completions_incident ON action_completions(incident_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_action_completions_completed ON action_completions(completed)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_category ON recommendations(category)")
        
        # Set schema version
        conn.execute("INSERT OR REPLACE INTO schema_version (version) VALUES (?)", (self._schema_version,))
    
    def store_spike_event(self, spike_event: SpikeEvent) -> int:
        """Store a spike event and return its ID."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO spike_events (
                        timestamp, endpoint, severity, baseline_latency, 
                        spike_latency, duration_seconds, affected_metrics, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    spike_event.timestamp.isoformat(),
                    spike_event.endpoint,
                    spike_event.severity.value,
                    spike_event.baseline_latency,
                    spike_event.spike_latency,
                    spike_event.duration.total_seconds(),
                    json.dumps(spike_event.affected_metrics),
                    json.dumps(spike_event.metadata)
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to store spike event: {e}")
            raise DatabaseError(f"Failed to store spike event: {e}")
    
    def store_correlation_result(self, correlation: CorrelationResult, spike_event_id: int) -> int:
        """Store a correlation result and return its ID."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO correlations (
                        spike_event_id, network_metrics, db_metrics, 
                        correlation_scores, confidence, analysis_timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    spike_event_id,
                    json.dumps(correlation.network_metrics),
                    json.dumps(correlation.db_metrics),
                    json.dumps(correlation.correlation_scores),
                    correlation.confidence,
                    correlation.analysis_timestamp.isoformat()
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to store correlation result: {e}")
            raise DatabaseError(f"Failed to store correlation result: {e}")
    
    def store_recommendation(self, root_cause: RootCause, correlation_id: int) -> int:
        """Store a recommendation and return its ID."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO recommendations (
                        correlation_id, category, description, confidence_score,
                        supporting_evidence, recommended_actions, priority
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    correlation_id,
                    root_cause.category,
                    root_cause.description,
                    root_cause.confidence_score,
                    json.dumps(root_cause.supporting_evidence),
                    json.dumps(root_cause.recommended_actions),
                    root_cause.priority
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to store recommendation: {e}")
            raise DatabaseError(f"Failed to store recommendation: {e}")
    
    def get_recent_spike_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent spike events with basic information."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM spike_events 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get recent spike events: {e}")
            raise DatabaseError(f"Failed to get recent spike events: {e}")
    
    def get_spike_event_with_analysis(self, spike_event_id: int) -> Optional[Dict[str, Any]]:
        """Get a spike event with its correlation and recommendations."""
        try:
            with self.get_connection() as conn:
                # Get spike event
                cursor = conn.execute("""
                    SELECT * FROM spike_events WHERE id = ?
                """, (spike_event_id,))
                spike_row = cursor.fetchone()
                
                if not spike_row:
                    return None
                
                spike_data = dict(spike_row)
                
                # Get correlation data
                cursor = conn.execute("""
                    SELECT * FROM correlations WHERE spike_event_id = ?
                """, (spike_event_id,))
                correlation_row = cursor.fetchone()
                
                if correlation_row:
                    spike_data['correlation'] = dict(correlation_row)
                    
                    # Get recommendations
                    cursor = conn.execute("""
                        SELECT * FROM recommendations 
                        WHERE correlation_id = ? 
                        ORDER BY priority ASC
                    """, (correlation_row['id'],))
                    spike_data['recommendations'] = [dict(row) for row in cursor.fetchall()]
                
                return spike_data
        except Exception as e:
            logger.error(f"Failed to get spike event with analysis: {e}")
            raise DatabaseError(f"Failed to get spike event with analysis: {e}")
    
    def update_recommendation_status(self, recommendation_id: int, status: str, feedback: Optional[str] = None) -> None:
        """Update recommendation status and feedback."""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    UPDATE recommendations 
                    SET status = ?, feedback = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (status, feedback, recommendation_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update recommendation status: {e}")
            raise DatabaseError(f"Failed to update recommendation status: {e}")
    
    def get_historical_patterns(self, category: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical incident patterns for knowledge base queries."""
        try:
            with self.get_connection() as conn:
                if category:
                    cursor = conn.execute("""
                        SELECT s.*, c.*, r.*
                        FROM spike_events s
                        JOIN correlations c ON s.id = c.spike_event_id
                        JOIN recommendations r ON c.id = r.correlation_id
                        WHERE r.category = ?
                        ORDER BY s.timestamp DESC
                        LIMIT ?
                    """, (category, limit))
                else:
                    cursor = conn.execute("""
                        SELECT s.*, c.*, r.*
                        FROM spike_events s
                        JOIN correlations c ON s.id = c.spike_event_id
                        JOIN recommendations r ON c.id = r.correlation_id
                        ORDER BY s.timestamp DESC
                        LIMIT ?
                    """, (limit,))
                
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get historical patterns: {e}")
            raise DatabaseError(f"Failed to get historical patterns: {e}")
    
    def get_schema_version(self) -> int:
        """Get current schema version from database."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
                row = cursor.fetchone()
                return row[0] if row else 0
        except sqlite3.OperationalError:
            # Table doesn't exist, this is a new database
            return 0
        except Exception as e:
            logger.error(f"Failed to get schema version: {e}")
            return 0
    
    def needs_migration(self) -> bool:
        """Check if database needs migration."""
        current_version = self.get_schema_version()
        return current_version < self._schema_version
    
    def migrate_schema(self) -> None:
        """Migrate database schema to current version."""
        current_version = self.get_schema_version()
        
        if current_version >= self._schema_version:
            logger.info(f"Database schema is up to date (version {current_version})")
            return
        
        logger.info(f"Migrating database schema from version {current_version} to {self._schema_version}")
        
        try:
            with self.get_connection() as conn:
                # For now, we only have version 1, so just create tables
                # In future versions, add migration logic here
                if current_version == 0:
                    self._create_tables(conn)
                    logger.info("Database schema migrated successfully")
                
                conn.commit()
        except Exception as e:
            logger.error(f"Schema migration failed: {e}")
            raise DatabaseError(f"Schema migration failed: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics and health information."""
        try:
            with self.get_connection() as conn:
                stats = {}
                
                # Get table counts
                cursor = conn.execute("SELECT COUNT(*) FROM spike_events")
                stats['spike_events_count'] = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM correlations")
                stats['correlations_count'] = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM recommendations")
                stats['recommendations_count'] = cursor.fetchone()[0]
                
                # Get database size
                cursor = conn.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                cursor = conn.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                stats['database_size_bytes'] = page_count * page_size
                
                # Get schema version
                stats['schema_version'] = self.get_schema_version()
                
                # Get recent activity
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM spike_events 
                    WHERE created_at > datetime('now', '-24 hours')
                """)
                stats['recent_spike_events'] = cursor.fetchone()[0]
                
                return stats
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {'error': str(e)}
    
    def health_check(self) -> bool:
        """Check if database is accessible and healthy."""
        try:
            with self.get_connection() as conn:
                conn.execute("SELECT 1")
                
                # Check if schema is up to date
                if self.needs_migration():
                    logger.warning("Database schema needs migration")
                    return False
                
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False