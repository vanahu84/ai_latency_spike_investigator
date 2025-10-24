"""
Gemini AI Engine for intelligent analysis and recommendation enhancement.

This module integrates with Google's Gemini API to provide AI-powered insights
for complex metric patterns and enhanced root cause analysis recommendations.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from config import config
from models.core import CorrelationResult, RootCause, Metric, BaseModel, ValidationError


class AIAnalysisType(Enum):
    """Types of AI analysis that can be performed."""
    PATTERN_ANALYSIS = "pattern_analysis"
    RECOMMENDATION_ENHANCEMENT = "recommendation_enhancement"
    ANOMALY_DETECTION = "anomaly_detection"


@dataclass
class AIInsight(BaseModel):
    """AI-generated insight from Gemini analysis."""
    
    analysis_type: AIAnalysisType
    confidence: float
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def validate(self) -> None:
        """Validate AI insight data."""
        if not isinstance(self.analysis_type, AIAnalysisType):
            raise ValidationError("analysis_type must be an AIAnalysisType enum")
        
        if not isinstance(self.confidence, (int, float)) or not (0 <= self.confidence <= 1):
            raise ValidationError("confidence must be between 0 and 1")
        
        if not isinstance(self.insights, list):
            raise ValidationError("insights must be a list")
        
        if not isinstance(self.recommendations, list):
            raise ValidationError("recommendations must be a list")
        
        if not isinstance(self.reasoning, str):
            raise ValidationError("reasoning must be a string")
        
        if not isinstance(self.metadata, dict):
            raise ValidationError("metadata must be a dictionary")
        
        if not isinstance(self.timestamp, datetime):
            raise ValidationError("timestamp must be a datetime object")
    
    def __post_init__(self):
        """Validate after initialization."""
        self.validate()


@dataclass
class EnhancedRecommendation(BaseModel):
    """Enhanced recommendation with AI insights."""
    
    original_cause: RootCause
    ai_insights: AIInsight
    enhanced_description: str
    enhanced_actions: List[str] = field(default_factory=list)
    combined_confidence: float = 0.0
    
    def validate(self) -> None:
        """Validate enhanced recommendation data."""
        if not isinstance(self.original_cause, RootCause):
            raise ValidationError("original_cause must be a RootCause instance")
        
        if not isinstance(self.ai_insights, AIInsight):
            raise ValidationError("ai_insights must be an AIInsight instance")
        
        if not self.enhanced_description or not isinstance(self.enhanced_description, str):
            raise ValidationError("enhanced_description must be a non-empty string")
        
        if not isinstance(self.enhanced_actions, list):
            raise ValidationError("enhanced_actions must be a list")
        
        if not isinstance(self.combined_confidence, (int, float)) or not (0 <= self.combined_confidence <= 1):
            raise ValidationError("combined_confidence must be between 0 and 1")
    
    def __post_init__(self):
        """Validate after initialization."""
        self.validate()


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire permission to make an API call."""
        async with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            if len(self.requests) >= self.max_requests:
                # Calculate wait time until oldest request expires
                wait_time = 60 - (now - self.requests[0]) + 0.1  # Add small buffer
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    # Clean up again after waiting
                    now = time.time()
                    self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            self.requests.append(now)


class GeminiAIEngine:
    """
    Gemini AI Engine for intelligent analysis and recommendation enhancement.
    
    This class provides AI-powered insights for complex metric patterns and
    enhances root cause analysis recommendations using Google's Gemini API.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-pro"):
        """
        Initialize the Gemini AI Engine.
        
        Args:
            api_key: Gemini API key (defaults to config value)
            model_name: Name of the Gemini model to use
        """
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.rate_limiter = RateLimiter(config.GEMINI_REQUESTS_PER_MINUTE)
        self.model = None
        self._is_available = False
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(
                    model_name=self.model_name,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    }
                )
                self._is_available = True
                self.logger.info(f"Gemini AI Engine initialized with model: {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize Gemini AI Engine: {e}")
                self._is_available = False
        else:
            self.logger.warning("No Gemini API key provided - AI features will be disabled")
    
    def is_available(self) -> bool:
        """Check if the Gemini AI Engine is available."""
        return self._is_available and self.model is not None
    
    async def analyze_complex_patterns(self, correlation_result: CorrelationResult) -> Optional[AIInsight]:
        """
        Analyze complex metric patterns using Gemini AI.
        
        Args:
            correlation_result: The correlation analysis result to analyze
            
        Returns:
            AI insights or None if analysis fails
        """
        if not self.is_available():
            self.logger.warning("Gemini AI Engine not available for pattern analysis")
            return None
        
        try:
            await self.rate_limiter.acquire()
            
            prompt = self._create_pattern_analysis_prompt(correlation_result)
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            
            if response and response.text:
                return self._parse_pattern_analysis_response(response.text)
            else:
                self.logger.warning("Empty response from Gemini API for pattern analysis")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in Gemini pattern analysis: {e}")
            return None
    
    async def enhance_recommendations(self, basic_rca: List[RootCause]) -> List[EnhancedRecommendation]:
        """
        Enhance basic RCA recommendations using AI insights.
        
        Args:
            basic_rca: List of basic root cause analysis results
            
        Returns:
            List of enhanced recommendations
        """
        if not self.is_available():
            self.logger.warning("Gemini AI Engine not available for recommendation enhancement")
            return [self._create_fallback_enhancement(cause) for cause in basic_rca]
        
        enhanced_recommendations = []
        
        for cause in basic_rca:
            try:
                await self.rate_limiter.acquire()
                
                prompt = self._create_enhancement_prompt(cause)
                
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    prompt
                )
                
                if response and response.text:
                    ai_insight = self._parse_enhancement_response(response.text)
                    if ai_insight:
                        enhanced_rec = self._create_enhanced_recommendation(cause, ai_insight)
                        enhanced_recommendations.append(enhanced_rec)
                    else:
                        enhanced_recommendations.append(self._create_fallback_enhancement(cause))
                else:
                    enhanced_recommendations.append(self._create_fallback_enhancement(cause))
                    
            except Exception as e:
                self.logger.error(f"Error enhancing recommendation for {cause.category}: {e}")
                enhanced_recommendations.append(self._create_fallback_enhancement(cause))
        
        return enhanced_recommendations
    
    def _create_pattern_analysis_prompt(self, correlation_result: CorrelationResult) -> str:
        """Create a prompt for pattern analysis."""
        spike = correlation_result.spike_event
        
        prompt = f"""
You are an expert system performance analyst. Analyze the following latency spike and correlation data to identify patterns and potential root causes.

SPIKE EVENT:
- Endpoint: {spike.endpoint}
- Baseline Latency: {spike.baseline_latency}ms
- Spike Latency: {spike.spike_latency}ms
- Spike Ratio: {spike.spike_ratio:.2f}x
- Duration: {spike.duration.total_seconds()}s
- Severity: {spike.severity.value}
- Timestamp: {spike.timestamp.isoformat()}

CORRELATION DATA:
Network Metrics: {json.dumps(correlation_result.network_metrics, indent=2)}
Database Metrics: {json.dumps(correlation_result.db_metrics, indent=2)}
Correlation Scores: {json.dumps(correlation_result.correlation_scores, indent=2)}
Overall Confidence: {correlation_result.confidence:.2f}

Please provide your analysis in the following JSON format:
{{
    "confidence": <float between 0 and 1>,
    "insights": [
        "<insight 1>",
        "<insight 2>",
        ...
    ],
    "recommendations": [
        "<recommendation 1>",
        "<recommendation 2>",
        ...
    ],
    "reasoning": "<detailed reasoning for your analysis>"
}}

Focus on:
1. Identifying unusual patterns in the correlation data
2. Suggesting potential root causes based on the metric relationships
3. Providing actionable insights for investigation
4. Considering the severity and duration of the spike
"""
        return prompt
    
    def _create_enhancement_prompt(self, root_cause: RootCause) -> str:
        """Create a prompt for recommendation enhancement."""
        prompt = f"""
You are an expert DevOps engineer. Enhance the following root cause analysis with more detailed and actionable recommendations.

ROOT CAUSE ANALYSIS:
- Category: {root_cause.category}
- Description: {root_cause.description}
- Confidence: {root_cause.confidence_score:.2f}
- Priority: {root_cause.priority}
- Supporting Evidence: {json.dumps(root_cause.supporting_evidence, indent=2)}
- Current Recommendations: {json.dumps(root_cause.recommended_actions, indent=2)}

Please provide enhanced analysis in the following JSON format:
{{
    "confidence": <float between 0 and 1>,
    "insights": [
        "<additional insight 1>",
        "<additional insight 2>",
        ...
    ],
    "recommendations": [
        "<enhanced recommendation 1>",
        "<enhanced recommendation 2>",
        ...
    ],
    "reasoning": "<reasoning for the enhancements>"
}}

Focus on:
1. Adding more specific and actionable steps
2. Providing troubleshooting commands or tools
3. Suggesting monitoring improvements
4. Including preventive measures
5. Considering the specific category ({root_cause.category}) context
"""
        return prompt
    
    def _parse_pattern_analysis_response(self, response_text: str) -> Optional[AIInsight]:
        """Parse the pattern analysis response from Gemini."""
        try:
            # Extract JSON from response (handle potential markdown formatting)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                self.logger.error("No JSON found in Gemini response")
                return None
            
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)
            
            return AIInsight(
                analysis_type=AIAnalysisType.PATTERN_ANALYSIS,
                confidence=float(data.get('confidence', 0.5)),
                insights=data.get('insights', []),
                recommendations=data.get('recommendations', []),
                reasoning=data.get('reasoning', ''),
                metadata={'raw_response': response_text}
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"Error parsing Gemini pattern analysis response: {e}")
            return None
    
    def _parse_enhancement_response(self, response_text: str) -> Optional[AIInsight]:
        """Parse the enhancement response from Gemini."""
        try:
            # Extract JSON from response (handle potential markdown formatting)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                self.logger.error("No JSON found in Gemini enhancement response")
                return None
            
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)
            
            return AIInsight(
                analysis_type=AIAnalysisType.RECOMMENDATION_ENHANCEMENT,
                confidence=float(data.get('confidence', 0.5)),
                insights=data.get('insights', []),
                recommendations=data.get('recommendations', []),
                reasoning=data.get('reasoning', ''),
                metadata={'raw_response': response_text}
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"Error parsing Gemini enhancement response: {e}")
            return None
    
    def _create_enhanced_recommendation(self, original_cause: RootCause, ai_insight: AIInsight) -> EnhancedRecommendation:
        """Create an enhanced recommendation combining original and AI insights."""
        # Combine original and AI recommendations
        enhanced_actions = list(original_cause.recommended_actions)
        enhanced_actions.extend(ai_insight.recommendations)
        
        # Create enhanced description
        enhanced_description = f"{original_cause.description}\n\nAI Analysis: {ai_insight.reasoning}"
        
        # Calculate combined confidence (weighted average)
        combined_confidence = (original_cause.confidence_score * 0.6) + (ai_insight.confidence * 0.4)
        
        return EnhancedRecommendation(
            original_cause=original_cause,
            ai_insights=ai_insight,
            enhanced_description=enhanced_description,
            enhanced_actions=enhanced_actions,
            combined_confidence=combined_confidence
        )
    
    def _create_fallback_enhancement(self, original_cause: RootCause) -> EnhancedRecommendation:
        """Create a fallback enhancement when AI is not available."""
        fallback_insight = AIInsight(
            analysis_type=AIAnalysisType.RECOMMENDATION_ENHANCEMENT,
            confidence=0.0,
            insights=["AI analysis not available - using rule-based recommendations"],
            recommendations=[],
            reasoning="Gemini AI Engine not available for enhancement"
        )
        
        return EnhancedRecommendation(
            original_cause=original_cause,
            ai_insights=fallback_insight,
            enhanced_description=original_cause.description,
            enhanced_actions=original_cause.recommended_actions,
            combined_confidence=original_cause.confidence_score
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the Gemini AI Engine.
        
        Returns:
            Health check results
        """
        health_status = {
            'service': 'gemini_ai_engine',
            'status': 'healthy' if self.is_available() else 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'details': {
                'api_key_configured': bool(self.api_key),
                'model_initialized': self.model is not None,
                'model_name': self.model_name,
                'rate_limit_requests_per_minute': config.GEMINI_REQUESTS_PER_MINUTE
            }
        }
        
        if self.is_available():
            try:
                # Test with a simple prompt
                await self.rate_limiter.acquire()
                test_response = await asyncio.to_thread(
                    self.model.generate_content,
                    "Respond with 'OK' if you can process this request."
                )
                
                if test_response and test_response.text:
                    health_status['details']['api_test'] = 'passed'
                else:
                    health_status['status'] = 'degraded'
                    health_status['details']['api_test'] = 'failed - empty response'
                    
            except Exception as e:
                health_status['status'] = 'degraded'
                health_status['details']['api_test'] = f'failed - {str(e)}'
        
        return health_status