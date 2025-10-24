"""
Core data models for the Latency Spike Root Cause Investigator.

This module contains the primary data structures used throughout the application
for representing metrics, spike events, correlation results, and root cause analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
import json
from abc import ABC, abstractmethod


class SeverityLevel(Enum):
    """Enumeration for spike severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class BaseModel(ABC):
    """Abstract base class for all data models with validation."""
    
    @abstractmethod
    def validate(self) -> None:
        """Validate the model data and raise ValidationError if invalid."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary for serialization."""
        if hasattr(self, '__dataclass_fields__'):
            result = {}
            for field_name, field_def in self.__dataclass_fields__.items():
                value = getattr(self, field_name)
                if isinstance(value, datetime):
                    result[field_name] = value.isoformat()
                elif isinstance(value, timedelta):
                    result[field_name] = value.total_seconds()
                elif isinstance(value, Enum):
                    result[field_name] = value.value
                elif isinstance(value, BaseModel):
                    result[field_name] = value.to_dict()
                elif isinstance(value, list) and value and isinstance(value[0], BaseModel):
                    result[field_name] = [item.to_dict() for item in value]
                else:
                    result[field_name] = value
            return result
        return {}
    
    def to_json(self) -> str:
        """Convert the model to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class Metric(BaseModel):
    """Represents a single metric data point with timestamp and metadata."""
    
    timestamp: datetime
    value: float
    metric_name: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate metric data."""
        if not isinstance(self.timestamp, datetime):
            raise ValidationError("timestamp must be a datetime object")
        
        if not isinstance(self.value, (int, float)):
            raise ValidationError("value must be a number")
        
        if not self.metric_name or not isinstance(self.metric_name, str):
            raise ValidationError("metric_name must be a non-empty string")
        
        if not self.source or not isinstance(self.source, str):
            raise ValidationError("source must be a non-empty string")
        
        if not isinstance(self.metadata, dict):
            raise ValidationError("metadata must be a dictionary")
    
    def __post_init__(self):
        """Validate after initialization."""
        self.validate()


@dataclass
class SpikeEvent(BaseModel):
    """Represents a detected latency spike with context and severity."""
    
    timestamp: datetime
    endpoint: str
    severity: SeverityLevel
    baseline_latency: float
    spike_latency: float
    duration: timedelta
    affected_metrics: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate spike event data."""
        if not isinstance(self.timestamp, datetime):
            raise ValidationError("timestamp must be a datetime object")
        
        if not self.endpoint or not isinstance(self.endpoint, str):
            raise ValidationError("endpoint must be a non-empty string")
        
        if not isinstance(self.severity, SeverityLevel):
            raise ValidationError("severity must be a SeverityLevel enum")
        
        if not isinstance(self.baseline_latency, (int, float)) or self.baseline_latency < 0:
            raise ValidationError("baseline_latency must be a non-negative number")
        
        if not isinstance(self.spike_latency, (int, float)) or self.spike_latency < 0:
            raise ValidationError("spike_latency must be a non-negative number")
        
        if self.spike_latency <= self.baseline_latency:
            raise ValidationError("spike_latency must be greater than baseline_latency")
        
        if not isinstance(self.duration, timedelta) or self.duration.total_seconds() < 0:
            raise ValidationError("duration must be a non-negative timedelta")
        
        if not isinstance(self.affected_metrics, list):
            raise ValidationError("affected_metrics must be a list")
        
        if not isinstance(self.metadata, dict):
            raise ValidationError("metadata must be a dictionary")
    
    def __post_init__(self):
        """Validate after initialization."""
        self.validate()
    
    @property
    def spike_ratio(self) -> float:
        """Calculate the ratio of spike latency to baseline latency."""
        return self.spike_latency / self.baseline_latency if self.baseline_latency > 0 else 0.0


@dataclass
class CorrelationResult(BaseModel):
    """Analysis results linking spike to other metrics."""
    
    spike_event: SpikeEvent
    network_metrics: Dict[str, float] = field(default_factory=dict)
    db_metrics: Dict[str, float] = field(default_factory=dict)
    correlation_scores: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    
    def validate(self) -> None:
        """Validate correlation result data."""
        if not isinstance(self.spike_event, SpikeEvent):
            raise ValidationError("spike_event must be a SpikeEvent instance")
        
        # Validate the spike event
        self.spike_event.validate()
        
        if not isinstance(self.network_metrics, dict):
            raise ValidationError("network_metrics must be a dictionary")
        
        if not isinstance(self.db_metrics, dict):
            raise ValidationError("db_metrics must be a dictionary")
        
        if not isinstance(self.correlation_scores, dict):
            raise ValidationError("correlation_scores must be a dictionary")
        
        # Validate correlation scores are between -1 and 1
        for metric, score in self.correlation_scores.items():
            if not isinstance(score, (int, float)) or not (-1 <= score <= 1):
                raise ValidationError(f"correlation score for {metric} must be between -1 and 1")
        
        if not isinstance(self.confidence, (int, float)) or not (0 <= self.confidence <= 1):
            raise ValidationError("confidence must be between 0 and 1")
        
        if not isinstance(self.analysis_timestamp, datetime):
            raise ValidationError("analysis_timestamp must be a datetime object")
    
    def __post_init__(self):
        """Validate after initialization."""
        self.validate()
    
    def get_strongest_correlations(self, threshold: float = 0.5) -> Dict[str, float]:
        """Get correlations above the specified threshold."""
        return {
            metric: score for metric, score in self.correlation_scores.items()
            if abs(score) >= threshold
        }


@dataclass
class RootCause(BaseModel):
    """Identified potential cause with confidence and recommendations."""
    
    category: str  # 'network', 'database', 'application', 'infrastructure'
    description: str
    confidence_score: float
    supporting_evidence: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    priority: int = 1  # 1 = highest priority
    
    VALID_CATEGORIES = {'network', 'database', 'application', 'infrastructure', 'external'}
    
    def validate(self) -> None:
        """Validate root cause data."""
        if not self.category or self.category not in self.VALID_CATEGORIES:
            raise ValidationError(f"category must be one of {self.VALID_CATEGORIES}")
        
        if not self.description or not isinstance(self.description, str):
            raise ValidationError("description must be a non-empty string")
        
        if not isinstance(self.confidence_score, (int, float)) or not (0 <= self.confidence_score <= 1):
            raise ValidationError("confidence_score must be between 0 and 1")
        
        if not isinstance(self.supporting_evidence, list):
            raise ValidationError("supporting_evidence must be a list")
        
        if not isinstance(self.recommended_actions, list):
            raise ValidationError("recommended_actions must be a list")
        
        if not isinstance(self.priority, int) or self.priority < 1:
            raise ValidationError("priority must be a positive integer")
    
    def __post_init__(self):
        """Validate after initialization."""
        self.validate()