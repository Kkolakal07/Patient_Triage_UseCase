"""
Priority Scoring Engine for Patient Incident Reports

Computes a composite priority score based on multiple factors:
- Severity of harm
- Likelihood of recurrence  
- Patient impact
- Regulatory/compliance risk
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta

from models import (
    ClassificationResult, PriorityScore, IncidentCategory,
    SeverityLevel, IncidentReport
)
from config import PRIORITY_WEIGHTS, SLA_BY_URGENCY


class PriorityScoringEngine:
    """
    Computes priority scores for incident reports based on classification results.
    """
    
    def __init__(self, historical_incidents: Optional[List[Dict]] = None):
        """
        Initialize with optional historical incident data for recurrence detection.
        
        Args:
            historical_incidents: List of past incidents for pattern detection
        """
        self.historical_incidents = historical_incidents or []
        self.weights = PRIORITY_WEIGHTS
    
    def compute_priority(
        self, 
        classification: ClassificationResult,
        report_text: str,
        department: Optional[str] = None,
        location: Optional[str] = None
    ) -> PriorityScore:
        """
        Compute comprehensive priority score.
        
        Returns PriorityScore with breakdown of components.
        """
        # Calculate individual components (each on 1-10 scale)
        severity_score = self._compute_severity_score(classification.severity)
        recurrence_score = self._compute_recurrence_score(
            classification.primary_category, department, location
        )
        impact_score = self._compute_patient_impact_score(
            classification, report_text
        )
        regulatory_score = self._compute_regulatory_score(classification)
        
        # Apply weights
        weighted_severity = severity_score * self.weights["severity"]
        weighted_recurrence = recurrence_score * self.weights["recurrence"]
        weighted_impact = impact_score * self.weights["patient_impact"]
        weighted_regulatory = regulatory_score * self.weights["regulatory"]
        
        # Total score (1-10 scale)
        total_score = (
            weighted_severity + 
            weighted_recurrence + 
            weighted_impact + 
            weighted_regulatory
        )
        total_score = round(min(max(total_score, 1), 10), 2)
        
        # Determine urgency level
        urgency = self._determine_urgency_level(total_score, classification.severity)
        
        # Get SLA
        sla_hours = SLA_BY_URGENCY.get(urgency, 72)
        
        return PriorityScore(
            total_score=total_score,
            severity_component=round(weighted_severity, 2),
            recurrence_component=round(weighted_recurrence, 2),
            patient_impact_component=round(weighted_impact, 2),
            regulatory_component=round(weighted_regulatory, 2),
            urgency_level=urgency,
            recommended_sla_hours=sla_hours
        )
    
    def _compute_severity_score(self, severity: SeverityLevel) -> float:
        """
        Convert severity level to 1-10 score.
        """
        severity_map = {
            SeverityLevel.NEAR_MISS: 2.0,
            SeverityLevel.MINOR_HARM: 4.0,
            SeverityLevel.MODERATE_HARM: 6.0,
            SeverityLevel.SERIOUS_HARM: 8.0,
            SeverityLevel.SENTINEL_EVENT: 10.0,
        }
        return severity_map.get(severity, 5.0)
    
    def _compute_recurrence_score(
        self,
        category: IncidentCategory,
        department: Optional[str],
        location: Optional[str]
    ) -> float:
        """
        Score based on likelihood of recurrence.
        Checks for similar past incidents.
        """
        if not self.historical_incidents:
            # No history available - assume moderate risk
            return 5.0
        
        # Count similar incidents in past 90 days
        similar_count = 0
        cutoff_date = datetime.now() - timedelta(days=90)
        
        for incident in self.historical_incidents:
            incident_date = incident.get("date")
            if incident_date and incident_date < cutoff_date:
                continue
            
            # Check for matches
            matches = 0
            if incident.get("category") == category.value:
                matches += 2
            if department and incident.get("department") == department:
                matches += 1
            if location and incident.get("location") == location:
                matches += 2  # Same location is significant
            
            if matches >= 2:
                similar_count += 1
        
        # Score based on recurrence pattern
        if similar_count >= 5:
            return 10.0  # Systemic issue
        elif similar_count >= 3:
            return 8.0   # Pattern emerging
        elif similar_count >= 1:
            return 6.0   # Has happened before
        else:
            return 3.0   # No recent similar incidents
    
    def _compute_patient_impact_score(
        self,
        classification: ClassificationResult,
        report_text: str
    ) -> float:
        """
        Score based on patient impact and vulnerability.
        """
        score = 5.0  # Baseline
        
        # Adjust based on sentiment (distress level)
        if classification.sentiment_score < -0.5:
            score += 2.0  # High distress
        elif classification.sentiment_score < 0:
            score += 1.0  # Moderate distress
        
        # Check for vulnerable populations
        vulnerable_indicators = [
            "child", "infant", "pediatric", "elderly", "pregnant",
            "disabled", "mental health", "dementia", "confused",
            "non-english", "homeless", "immunocompromised"
        ]
        
        text_lower = report_text.lower()
        vulnerable_count = sum(1 for ind in vulnerable_indicators if ind in text_lower)
        score += min(vulnerable_count, 2)  # Max +2 for vulnerable
        
        # Check for multiple patients affected
        if any(phrase in text_lower for phrase in ["multiple patients", "several patients", "other patients"]):
            score += 2.0
        
        return min(score, 10.0)
    
    def _compute_regulatory_score(self, classification: ClassificationResult) -> float:
        """
        Score based on regulatory/compliance risk.
        """
        # High regulatory concern categories
        high_risk_categories = [
            IncidentCategory.SENTINEL_EVENT if hasattr(IncidentCategory, 'SENTINEL_EVENT') else None,
            IncidentCategory.PRIVACY_BREACH,
            IncidentCategory.SURGICAL_ERROR,
            IncidentCategory.MEDICATION_ERROR,
            IncidentCategory.INFECTION,
        ]
        
        medium_risk_categories = [
            IncidentCategory.DIAGNOSIS_ERROR,
            IncidentCategory.PATIENT_FALL,
            IncidentCategory.EQUIPMENT_FAILURE,
        ]
        
        category = classification.primary_category
        
        # Check severity for sentinel events (always high regulatory risk)
        if classification.severity == SeverityLevel.SENTINEL_EVENT:
            return 10.0
        
        if category in high_risk_categories:
            return 8.0
        elif category in medium_risk_categories:
            return 6.0
        else:
            return 4.0
    
    def _determine_urgency_level(
        self, 
        total_score: float, 
        severity: SeverityLevel
    ) -> str:
        """
        Determine urgency level from score and severity.
        """
        # Sentinel events are always critical
        if severity == SeverityLevel.SENTINEL_EVENT:
            return "CRITICAL"
        
        # Score-based determination
        if total_score >= 8.5:
            return "CRITICAL"
        elif total_score >= 6.5:
            return "HIGH"
        elif total_score >= 4.0:
            return "MEDIUM"
        else:
            return "LOW"
    
    def update_historical_data(self, incidents: List[Dict]):
        """Update historical incident data for recurrence detection"""
        self.historical_incidents.extend(incidents)
        
        # Keep only last 90 days
        cutoff = datetime.now() - timedelta(days=90)
        self.historical_incidents = [
            i for i in self.historical_incidents
            if i.get("date", datetime.now()) >= cutoff
        ]


class AdaptivePriorityScorer(PriorityScoringEngine):
    """
    Advanced priority scorer that learns from feedback.
    Adjusts weights based on historical accuracy.
    """
    
    def __init__(self, historical_incidents: Optional[List[Dict]] = None):
        super().__init__(historical_incidents)
        self.feedback_data = []
        self.weight_adjustments = {
            "severity": 1.0,
            "recurrence": 1.0,
            "patient_impact": 1.0,
            "regulatory": 1.0,
        }
    
    def record_feedback(
        self,
        incident_id: str,
        original_priority: PriorityScore,
        actual_urgency: str,
        was_accurate: bool
    ):
        """
        Record feedback from human reviewers to improve scoring.
        """
        self.feedback_data.append({
            "incident_id": incident_id,
            "predicted_urgency": original_priority.urgency_level,
            "actual_urgency": actual_urgency,
            "was_accurate": was_accurate,
            "timestamp": datetime.now()
        })
        
        # Periodically recalibrate
        if len(self.feedback_data) % 100 == 0:
            self._recalibrate_weights()
    
    def _recalibrate_weights(self):
        """
        Adjust weights based on accumulated feedback.
        (Simplified version - production would use ML)
        """
        if len(self.feedback_data) < 50:
            return  # Not enough data
        
        # Calculate accuracy rates
        recent_feedback = self.feedback_data[-100:]
        accuracy = sum(1 for f in recent_feedback if f["was_accurate"]) / len(recent_feedback)
        
        # If accuracy is low, suggest review
        if accuracy < 0.8:
            print(f"Warning: Priority scoring accuracy at {accuracy:.1%}. Consider model retraining.")
