"""
Automated Response Engine for COA-ROE Integration

This module provides real-time automated response capabilities based on timeline events
and threat levels, implementing the COA-ROE (Course of Action - Rules of Engagement) system.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Awaitable

import logging
# Use standard logging instead of FLAGSHIP.common
get_logger = logging.getLogger
from ....fusion.event_bus.event_bus_convoy import EventBusConvoy, Event

logger = get_logger(__name__)


class ThreatLevel(Enum):
    """Threat level enumeration for automated responses."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ResponseType(Enum):
    """Types of automated responses."""
    ALERT = "alert"
    NOTIFICATION = "notification"
    AUTOMATED_ACTION = "automated_action"
    ESCALATION = "escalation"
    ISOLATION = "isolation"
    COUNTERMEASURE = "countermeasure"


class ResponseStatus(Enum):
    """Status of response execution."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ResponseRule:
    """Rule definition for automated responses."""
    id: str
    name: str
    description: str
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    priority: int = 1
    enabled: bool = True
    cooldown_seconds: int = 0
    last_executed: Optional[datetime] = None
    execution_count: int = 0
    max_executions: Optional[int] = None


@dataclass
class ResponseAction:
    """Action to be executed as part of a response."""
    id: str
    type: ResponseType
    name: str
    description: str
    parameters: Dict[str, Any]
    timeout_seconds: int = 30
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class ResponseExecution:
    """Execution record for a response."""
    id: str
    rule_id: str
    event_id: str
    threat_level: ThreatLevel
    status: ResponseStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    actions_executed: List[str] = field(default_factory=list)
    actions_failed: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutomatedResponseEngine:
    """Engine for automated responses based on timeline events and threat levels."""

    def __init__(self, event_bus_convoy: EventBusConvoy, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Automated Response Engine.
        
        Args:
            event_bus_convoy: The event bus convoy for event processing
            config: Configuration dictionary
        """
        self.event_bus_convoy = event_bus_convoy
        self.config = config or {}
        
        # Response rules and actions
        self.response_rules: List[ResponseRule] = []
        self.response_actions: Dict[str, ResponseAction] = {}
        self.active_executions: Dict[str, ResponseExecution] = {}
        
        # Performance metrics
        self.metrics = {
            "total_responses": 0,
            "successful_responses": 0,
            "failed_responses": 0,
            "average_response_time": 0.0,
            "active_executions": 0,
        }
        
        # Event processing
        self._running = False
        self._event_processor_task: Optional[asyncio.Task] = None
        self._rule_evaluator_task: Optional[asyncio.Task] = None
        
        # Callbacks for external integrations
        self._response_callbacks: List[Callable[[ResponseExecution], Awaitable[None]]] = []
        
        # Initialize default rules
        self._initialize_default_rules()

    def _initialize_default_rules(self) -> None:
        """Initialize default response rules for common scenarios."""
        default_rules = [
            ResponseRule(
                id="high_threat_alert",
                name="High Threat Alert",
                description="Automatically alert operators for high threat events",
                conditions={
                    "threat_level": ["high", "critical"],
                    "score_impact": {"min": 15},
                    "source": ["video", "audio", "sensor"]
                },
                actions=[
                    {
                        "type": "alert",
                        "name": "High Threat Notification",
                        "parameters": {
                            "priority": "high",
                            "channels": ["dashboard", "email", "sms"]
                        }
                    }
                ],
                priority=1
            ),
            ResponseRule(
                id="correlation_escalation",
                name="Correlation Escalation",
                description="Escalate events with high correlation scores",
                conditions={
                    "correlation_strength": {"min": 0.8},
                    "correlation_count": {"min": 3}
                },
                actions=[
                    {
                        "type": "escalation",
                        "name": "Correlation Escalation",
                        "parameters": {
                            "level": "senior_analyst",
                            "urgency": "immediate"
                        }
                    }
                ],
                priority=2
            ),
            ResponseRule(
                id="rapid_response",
                name="Rapid Response",
                description="Immediate response for critical events",
                conditions={
                    "threat_level": "critical",
                    "response_time": {"max": 30}  # seconds
                },
                actions=[
                    {
                        "type": "automated_action",
                        "name": "Emergency Protocol",
                        "parameters": {
                            "protocol": "emergency_lockdown",
                            "scope": "affected_zones"
                        }
                    }
                ],
                priority=0  # Highest priority
            )
        ]
        
        for rule in default_rules:
            self.add_response_rule(rule)

    def add_response_rule(self, rule: ResponseRule) -> None:
        """Add a new response rule."""
        # Remove existing rule with same ID
        self.response_rules = [r for r in self.response_rules if r.id != rule.id]
        self.response_rules.append(rule)
        
        # Sort by priority (lower number = higher priority)
        self.response_rules.sort(key=lambda r: r.priority)
        logger.info(f"Added response rule: {rule.name} (priority: {rule.priority})")

    def add_response_action(self, action: ResponseAction) -> None:
        """Add a new response action."""
        self.response_actions[action.id] = action
        logger.info(f"Added response action: {action.name}")

    def add_response_callback(self, callback: Callable[[ResponseExecution], Awaitable[None]]) -> None:
        """Add a callback for response execution notifications."""
        self._response_callbacks.append(callback)

    async def start(self) -> None:
        """Start the automated response engine."""
        if self._running:
            return
            
        self._running = True
        
        # Subscribe to timeline events
        self.event_bus_convoy.subscribe_all(self._on_event_received)
        
        # Start background tasks
        self._event_processor_task = asyncio.create_task(self._event_processor_loop())
        self._rule_evaluator_task = asyncio.create_task(self._rule_evaluator_loop())
        
        logger.info("Automated Response Engine started")

    async def stop(self) -> None:
        """Stop the automated response engine."""
        if not self._running:
            return
            
        self._running = False
        
        # Cancel background tasks
        if self._event_processor_task:
            self._event_processor_task.cancel()
        if self._rule_evaluator_task:
            self._rule_evaluator_task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(
            self._event_processor_task, 
            self._rule_evaluator_task, 
            return_exceptions=True
        )
        
        logger.info("Automated Response Engine stopped")

    def _on_event_received(self, event: Event) -> None:
        """Handle incoming events from the timeline."""
        try:
            # Extract threat information from event
            threat_level = self._extract_threat_level(event)
            score_impact = getattr(event, 'scoreImpact', 0)
            source = getattr(event, 'source', 'unknown')
            
            # Create event context for rule evaluation
            event_context = {
                "event_id": event.id,
                "threat_level": threat_level.value,
                "score_impact": score_impact,
                "source": source,
                "timestamp": event.timestamp,
                "function": getattr(event, 'function', ''),
                "description": getattr(event, 'description', ''),
                "metadata": getattr(event, 'metadata', {})
            }
            
            # Queue event for rule evaluation
            asyncio.create_task(self._evaluate_event_rules(event_context))
            
        except Exception as e:
            logger.error(f"Error processing event {event.id}: {e}")

    def _extract_threat_level(self, event: Event) -> ThreatLevel:
        """Extract threat level from event data."""
        score_impact = getattr(event, 'scoreImpact', 0)
        
        if score_impact >= 25:
            return ThreatLevel.CRITICAL
        elif score_impact >= 15:
            return ThreatLevel.HIGH
        elif score_impact >= 8:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW

    async def _evaluate_event_rules(self, event_context: Dict[str, Any]) -> None:
        """Evaluate response rules for an event."""
        try:
            for rule in self.response_rules:
                if not rule.enabled:
                    continue
                    
                # Check cooldown
                if rule.cooldown_seconds > 0 and rule.last_executed:
                    time_since_execution = (datetime.now() - rule.last_executed).total_seconds()
                    if time_since_execution < rule.cooldown_seconds:
                        continue
                
                # Check max executions
                if rule.max_executions and rule.execution_count >= rule.max_executions:
                    continue
                
                # Evaluate rule conditions
                if self._evaluate_rule_conditions(rule.conditions, event_context):
                    # Execute response
                    await self._execute_response(rule, event_context)
                    break  # Only execute one rule per event (highest priority)
                    
        except Exception as e:
            logger.error(f"Error evaluating rules for event {event_context['event_id']}: {e}")

    def _evaluate_rule_conditions(self, conditions: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate if rule conditions are met."""
        try:
            for key, condition in conditions.items():
                if key not in context:
                    return False
                
                context_value = context[key]
                
                if isinstance(condition, list):
                    # Check if context value is in the list
                    if context_value not in condition:
                        return False
                elif isinstance(condition, dict):
                    # Range-based conditions
                    if "min" in condition and context_value < condition["min"]:
                        return False
                    if "max" in condition and context_value > condition["max"]:
                        return False
                else:
                    # Direct equality check
                    if context_value != condition:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating conditions: {e}")
            return False

    async def _execute_response(self, rule: ResponseRule, event_context: Dict[str, Any]) -> None:
        """Execute the response for a rule."""
        try:
            # Create execution record
            execution = ResponseExecution(
                id=f"exec_{int(time.time() * 1000)}",
                rule_id=rule.id,
                event_id=event_context["event_id"],
                threat_level=ThreatLevel(event_context["threat_level"]),
                status=ResponseStatus.EXECUTING,
                start_time=datetime.now()
            )
            
            self.active_executions[execution.id] = execution
            self.metrics["active_executions"] += 1
            self.metrics["total_responses"] += 1
            
            # Update rule execution count
            rule.execution_count += 1
            rule.last_executed = datetime.now()
            
            logger.info(f"Executing response rule: {rule.name} for event {event_context['event_id']}")
            
            # Execute actions
            for action_config in rule.actions:
                try:
                    await self._execute_action(action_config, execution)
                    execution.actions_executed.append(action_config["name"])
                except Exception as e:
                    execution.actions_failed.append(action_config["name"])
                    logger.error(f"Action execution failed: {action_config['name']} - {e}")
            
            # Complete execution
            execution.end_time = datetime.now()
            execution.status = ResponseStatus.COMPLETED
            
            # Update metrics
            response_time = (execution.end_time - execution.start_time).total_seconds()
            self.metrics["average_response_time"] = (
                (self.metrics["average_response_time"] * (self.metrics["total_responses"] - 1) + response_time) /
                self.metrics["total_responses"]
            )
            
            if not execution.actions_failed:
                self.metrics["successful_responses"] += 1
            else:
                self.metrics["failed_responses"] += 1
            
            self.metrics["active_executions"] -= 1
            
            # Notify callbacks
            await self._notify_response_callbacks(execution)
            
            # Clean up completed execution
            if execution.id in self.active_executions:
                del self.active_executions[execution.id]
                
        except Exception as e:
            logger.error(f"Error executing response for rule {rule.id}: {e}")
            if execution:
                execution.status = ResponseStatus.FAILED
                execution.error_message = str(e)
                execution.end_time = datetime.now()

    async def _execute_action(self, action_config: Dict[str, Any], execution: ResponseExecution) -> None:
        """Execute a specific action."""
        action_type = action_config["type"]
        action_name = action_config["name"]
        parameters = action_config.get("parameters", {})
        
        logger.info(f"Executing action: {action_name} (type: {action_type})")
        
        if action_type == "alert":
            await self._execute_alert_action(parameters, execution)
        elif action_type == "escalation":
            await self._execute_escalation_action(parameters, execution)
        elif action_type == "automated_action":
            await self._execute_automated_action(parameters, execution)
        elif action_type == "notification":
            await self._execute_notification_action(parameters, execution)
        else:
            logger.warning(f"Unknown action type: {action_type}")

    async def _execute_alert_action(self, parameters: Dict[str, Any], execution: ResponseExecution) -> None:
        """Execute an alert action."""
        priority = parameters.get("priority", "medium")
        channels = parameters.get("channels", ["dashboard"])
        
        # Create alert message
        alert_message = {
            "type": "automated_alert",
            "priority": priority,
            "message": f"Automated alert triggered by rule execution {execution.rule_id}",
            "event_id": execution.event_id,
            "threat_level": execution.threat_level.value,
            "timestamp": datetime.now().isoformat(),
            "channels": channels
        }
        
        # Publish alert to event bus
        alert_event = Event(
            id=f"alert_{execution.id}",
            type="automated_alert",
            source="automated_response_engine",
            timestamp=datetime.now().isoformat(),
            metadata=alert_message
        )
        
        self.event_bus_convoy.publish(alert_event)
        logger.info(f"Alert action executed: {alert_message}")

    async def _execute_escalation_action(self, parameters: Dict[str, Any], execution: ResponseExecution) -> None:
        """Execute an escalation action."""
        level = parameters.get("level", "analyst")
        urgency = parameters.get("urgency", "normal")
        
        escalation_message = {
            "type": "escalation",
            "level": level,
            "urgency": urgency,
            "event_id": execution.event_id,
            "rule_id": execution.rule_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Publish escalation event
        escalation_event = Event(
            id=f"escalation_{execution.id}",
            type="escalation",
            source="automated_response_engine",
            timestamp=datetime.now().isoformat(),
            metadata=escalation_message
        )
        
        self.event_bus_convoy.publish(escalation_event)
        logger.info(f"Escalation action executed: {escalation_message}")

    async def _execute_automated_action(self, parameters: Dict[str, Any], execution: ResponseExecution) -> None:
        """Execute an automated action."""
        protocol = parameters.get("protocol", "")
        scope = parameters.get("scope", "all")
        
        action_message = {
            "type": "automated_action",
            "protocol": protocol,
            "scope": scope,
            "event_id": execution.event_id,
            "rule_id": execution.rule_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Publish automated action event
        action_event = Event(
            id=f"action_{execution.id}",
            type="automated_action",
            source="automated_response_engine",
            timestamp=datetime.now().isoformat(),
            metadata=action_message
        )
        
        self.event_bus_convoy.publish(action_event)
        logger.info(f"Automated action executed: {action_message}")

    async def _execute_notification_action(self, parameters: Dict[str, Any], execution: ResponseExecution) -> None:
        """Execute a notification action."""
        recipients = parameters.get("recipients", [])
        message = parameters.get("message", "Automated notification")
        
        notification_message = {
            "type": "notification",
            "recipients": recipients,
            "message": message,
            "event_id": execution.event_id,
            "rule_id": execution.rule_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Publish notification event
        notification_event = Event(
            id=f"notification_{execution.id}",
            type="notification",
            source="automated_response_engine",
            timestamp=datetime.now().isoformat(),
            metadata=notification_message
        )
        
        self.event_bus_convoy.publish(notification_event)
        logger.info(f"Notification action executed: {notification_message}")

    async def _notify_response_callbacks(self, execution: ResponseExecution) -> None:
        """Notify external callbacks about response execution."""
        for callback in self._response_callbacks:
            try:
                await callback(execution)
            except Exception as e:
                logger.error(f"Error in response callback: {e}")

    async def _event_processor_loop(self) -> None:
        """Background loop for processing events."""
        while self._running:
            try:
                await asyncio.sleep(0.1)  # Process events every 100ms
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event processor loop: {e}")

    async def _rule_evaluator_loop(self) -> None:
        """Background loop for rule evaluation."""
        while self._running:
            try:
                await asyncio.sleep(1.0)  # Evaluate rules every second
                
                # Clean up old executions
                current_time = datetime.now()
                expired_executions = []
                
                for execution_id, execution in self.active_executions.items():
                    if execution.start_time < current_time - timedelta(minutes=30):  # 30 minute timeout
                        expired_executions.append(execution_id)
                
                for execution_id in expired_executions:
                    if execution_id in self.active_executions:
                        del self.active_executions[execution_id]
                        self.metrics["active_executions"] -= 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in rule evaluator loop: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get engine performance metrics."""
        return {
            **self.metrics,
            "active_rules": len([r for r in self.response_rules if r.enabled]),
            "total_rules": len(self.response_rules),
            "active_executions_count": len(self.active_executions),
            "response_rules": [
                {
                    "id": rule.id,
                    "name": rule.name,
                    "priority": rule.priority,
                    "enabled": rule.enabled,
                    "execution_count": rule.execution_count,
                    "last_executed": rule.last_executed.isoformat() if rule.last_executed else None
                }
                for rule in self.response_rules
            ]
        }

    def get_active_executions(self) -> List[ResponseExecution]:
        """Get list of currently active executions."""
        return list(self.active_executions.values())

    def get_response_rules(self) -> List[ResponseRule]:
        """Get list of response rules."""
        return self.response_rules.copy()
