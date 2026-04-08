"""
Routing Engine for Patient Incident Reports

Routes classified and prioritized incidents to appropriate personnel
based on rules, escalation paths, and notification preferences.
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta

from models import (
    ClassificationResult, PriorityScore, RoutingDecision,
    IncidentCategory, SeverityLevel, RouteDestination
)
from config import ROUTING_RULES


class RoutingEngine:
    """
    Routes incidents to appropriate personnel based on classification and priority.
    """
    
    def __init__(self, custom_rules: Optional[Dict] = None):
        """
        Initialize with optional custom routing rules.
        """
        self.rules = custom_rules or ROUTING_RULES
        self.escalation_timers = {}
    
    def route(
        self,
        classification: ClassificationResult,
        priority: PriorityScore,
        incident_id: str
    ) -> RoutingDecision:
        """
        Determine routing for an incident.
        
        Returns RoutingDecision with primary/secondary assignees and escalation path.
        """
        # Check for sentinel events first (highest priority routing)
        if classification.severity == SeverityLevel.SENTINEL_EVENT:
            return self._route_sentinel_event(priority)
        
        # Get routing rule based on category
        rule = self._get_routing_rule(classification.primary_category)
        
        # Build assignees
        primary = self._get_destination(rule["primary"])
        secondary = [self._get_destination(s) for s in rule.get("secondary", [])]
        
        # Build escalation path based on priority
        escalation = self._build_escalation_path(classification, priority)
        
        # Determine notification channels based on urgency
        notifications = self._determine_notifications(priority, rule)
        
        # Requires immediate attention?
        immediate = (
            rule.get("immediate", False) or 
            priority.urgency_level == "CRITICAL"
        )
        
        return RoutingDecision(
            primary_assignee=primary,
            secondary_assignees=secondary,
            escalation_path=escalation,
            requires_immediate_attention=immediate,
            notification_channels=notifications
        )
    
    def _route_sentinel_event(self, priority: PriorityScore) -> RoutingDecision:
        """
        Special routing for sentinel events - immediate executive notification.
        """
        rule = self.rules.get("sentinel_event", self.rules["default"])
        
        return RoutingDecision(
            primary_assignee=RouteDestination.CHIEF_MEDICAL_OFFICER,
            secondary_assignees=[
                RouteDestination.RISK_MANAGEMENT,
                RouteDestination.LEGAL,
                RouteDestination.COMPLIANCE
            ],
            escalation_path=[
                RouteDestination.CHIEF_MEDICAL_OFFICER,
                RouteDestination.RISK_MANAGEMENT,
            ],
            requires_immediate_attention=True,
            notification_channels=["pager", "sms", "email", "dashboard"]
        )
    
    def _get_routing_rule(self, category: IncidentCategory) -> Dict:
        """
        Get routing rule for a category.
        """
        # Map category to rule key
        category_to_rule = {
            IncidentCategory.MEDICATION_ERROR: "medication_error",
            IncidentCategory.INFECTION: "infection",
            IncidentCategory.EQUIPMENT_FAILURE: "equipment_failure",
            IncidentCategory.PATIENT_EXPERIENCE: "patient_experience",
            IncidentCategory.PRIVACY_BREACH: "privacy_breach",
            IncidentCategory.STAFF_CONDUCT: "staff_conduct",
        }
        
        rule_key = category_to_rule.get(category, "default")
        return self.rules.get(rule_key, self.rules["default"])
    
    def _get_destination(self, destination_name: str) -> RouteDestination:
        """
        Convert string destination to enum.
        """
        try:
            return RouteDestination(destination_name)
        except ValueError:
            # Try uppercase
            try:
                return RouteDestination[destination_name.upper()]
            except KeyError:
                return RouteDestination.QUALITY_IMPROVEMENT
    
    def _build_escalation_path(
        self,
        classification: ClassificationResult,
        priority: PriorityScore
    ) -> List[RouteDestination]:
        """
        Build escalation path based on severity and category.
        """
        path = []
        
        # Always include department head
        path.append(RouteDestination.DEPARTMENT_HEAD)
        
        # Add based on priority
        if priority.urgency_level in ["CRITICAL", "HIGH"]:
            path.append(RouteDestination.RISK_MANAGEMENT)
        
        if priority.urgency_level == "CRITICAL":
            path.append(RouteDestination.CHIEF_MEDICAL_OFFICER)
        
        # Category-specific escalation
        if classification.primary_category == IncidentCategory.MEDICATION_ERROR:
            path.insert(0, RouteDestination.PHARMACY_DIRECTOR)
        elif classification.primary_category == IncidentCategory.INFECTION:
            path.insert(0, RouteDestination.INFECTION_CONTROL)
        
        return path
    
    def _determine_notifications(
        self,
        priority: PriorityScore,
        rule: Dict
    ) -> List[str]:
        """
        Determine notification channels based on urgency.
        """
        base_notifications = rule.get("notification", ["email", "dashboard"])
        
        if priority.urgency_level == "CRITICAL":
            # Add immediate channels
            return list(set(base_notifications + ["pager", "sms"]))
        elif priority.urgency_level == "HIGH":
            return list(set(base_notifications + ["sms"]))
        else:
            return base_notifications
    
    def check_escalation(
        self,
        incident_id: str,
        current_assignee: RouteDestination,
        assigned_at: datetime,
        sla_hours: int
    ) -> Optional[RouteDestination]:
        """
        Check if incident needs escalation due to SLA breach.
        
        Returns next escalation destination if SLA breached, None otherwise.
        """
        time_elapsed = datetime.now() - assigned_at
        sla_deadline = timedelta(hours=sla_hours)
        
        if time_elapsed > sla_deadline:
            # SLA breached - return escalation destination
            escalation_map = {
                RouteDestination.UNIT_MANAGER: RouteDestination.DEPARTMENT_HEAD,
                RouteDestination.DEPARTMENT_HEAD: RouteDestination.RISK_MANAGEMENT,
                RouteDestination.PHARMACY_DIRECTOR: RouteDestination.CHIEF_MEDICAL_OFFICER,
                RouteDestination.PATIENT_RELATIONS: RouteDestination.DEPARTMENT_HEAD,
                RouteDestination.SAFETY_OFFICER: RouteDestination.RISK_MANAGEMENT,
            }
            return escalation_map.get(current_assignee, RouteDestination.RISK_MANAGEMENT)
        
        return None


class NotificationService:
    """
    Handles sending notifications through various channels.
    (Stub implementation - would integrate with actual notification systems)
    """
    
    def __init__(self):
        self.notification_log = []
    
    def send_notification(
        self,
        recipient: RouteDestination,
        incident_id: str,
        priority: PriorityScore,
        channels: List[str],
        message: str
    ) -> bool:
        """
        Send notification through specified channels.
        """
        notification = {
            "recipient": recipient.value,
            "incident_id": incident_id,
            "urgency": priority.urgency_level,
            "channels": channels,
            "message": message,
            "sent_at": datetime.now(),
            "status": "sent"
        }
        
        # Log the notification (in production, would actually send)
        self.notification_log.append(notification)
        
        # Simulate sending
        for channel in channels:
            self._send_via_channel(channel, recipient, message)
        
        return True
    
    def _send_via_channel(self, channel: str, recipient: RouteDestination, message: str):
        """
        Send via specific channel (stub implementation).
        """
        # In production, would integrate with:
        # - Email: SMTP/SendGrid/SES
        # - SMS: Twilio
        # - Pager: PagerDuty
        # - Dashboard: WebSocket push
        
        print(f"[{channel.upper()}] → {recipient.value}: {message[:50]}...")


class WorkflowEngine:
    """
    Manages the complete workflow for incident processing.
    """
    
    def __init__(self):
        self.router = RoutingEngine()
        self.notifier = NotificationService()
        self.active_incidents = {}
    
    def process_routing(
        self,
        incident_id: str,
        classification: ClassificationResult,
        priority: PriorityScore,
        report_summary: str
    ) -> RoutingDecision:
        """
        Process routing and send notifications.
        """
        # Get routing decision
        routing = self.router.route(classification, priority, incident_id)
        
        # Track active incident
        self.active_incidents[incident_id] = {
            "routing": routing,
            "assigned_at": datetime.now(),
            "priority": priority,
            "status": "assigned"
        }
        
        # Send notifications
        message = self._build_notification_message(
            incident_id, classification, priority, report_summary
        )
        
        # Notify primary
        self.notifier.send_notification(
            routing.primary_assignee,
            incident_id,
            priority,
            routing.notification_channels,
            message
        )
        
        # Notify secondary (if critical)
        if routing.requires_immediate_attention:
            for secondary in routing.secondary_assignees:
                self.notifier.send_notification(
                    secondary,
                    incident_id,
                    priority,
                    ["email"],
                    f"[CC] {message}"
                )
        
        return routing
    
    def _build_notification_message(
        self,
        incident_id: str,
        classification: ClassificationResult,
        priority: PriorityScore,
        summary: str
    ) -> str:
        """
        Build notification message.
        """
        return (
            f"[{priority.urgency_level}] Incident #{incident_id}\n"
            f"Category: {classification.primary_category.value}\n"
            f"Severity: {classification.severity.name}\n"
            f"Priority Score: {priority.total_score}/10\n"
            f"SLA: {priority.recommended_sla_hours} hours\n"
            f"Summary: {summary[:200]}..."
        )
    
    def check_sla_compliance(self) -> List[str]:
        """
        Check all active incidents for SLA compliance.
        Returns list of escalated incident IDs.
        """
        escalated = []
        
        for incident_id, data in self.active_incidents.items():
            if data["status"] != "assigned":
                continue
            
            escalation = self.router.check_escalation(
                incident_id,
                data["routing"].primary_assignee,
                data["assigned_at"],
                data["priority"].recommended_sla_hours
            )
            
            if escalation:
                # Trigger escalation
                self.notifier.send_notification(
                    escalation,
                    incident_id,
                    data["priority"],
                    ["sms", "email"],
                    f"[ESCALATION] SLA breach for incident #{incident_id}"
                )
                data["status"] = "escalated"
                escalated.append(incident_id)
        
        return escalated
