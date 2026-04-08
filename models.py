"""
Data models for the Patient Complaint & Incident Report Triage System
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class SeverityLevel(Enum):
    """Severity classification based on patient safety standards"""
    NEAR_MISS = 1          # No harm reached patient
    MINOR_HARM = 2         # Temporary harm, no intervention needed
    MODERATE_HARM = 3      # Temporary harm, intervention required
    SERIOUS_HARM = 4       # Permanent harm or prolonged hospitalization
    SENTINEL_EVENT = 5     # Death or serious permanent harm


class IncidentCategory(Enum):
    """Primary incident categories"""
    MEDICATION_ERROR = "medication_error"
    PATIENT_FALL = "patient_fall"
    INFECTION = "infection"
    EQUIPMENT_FAILURE = "equipment_failure"
    COMMUNICATION = "communication"
    DIAGNOSIS_ERROR = "diagnosis_error"
    SURGICAL_ERROR = "surgical_error"
    PATIENT_EXPERIENCE = "patient_experience"
    DOCUMENTATION = "documentation"
    PRIVACY_BREACH = "privacy_breach"
    STAFF_CONDUCT = "staff_conduct"
    DELAY_IN_CARE = "delay_in_care"
    OTHER = "other"


class Department(Enum):
    """Hospital departments"""
    EMERGENCY = "emergency"
    SURGERY = "surgery"
    ICU = "icu"
    RADIOLOGY = "radiology"
    PHARMACY = "pharmacy"
    NURSING = "nursing"
    LABORATORY = "laboratory"
    ADMINISTRATION = "administration"
    OUTPATIENT = "outpatient"
    PEDIATRICS = "pediatrics"
    MATERNITY = "maternity"
    CARDIOLOGY = "cardiology"
    ONCOLOGY = "oncology"
    GENERAL = "general"


class RouteDestination(Enum):
    """Routing destinations"""
    CHIEF_MEDICAL_OFFICER = "chief_medical_officer"
    RISK_MANAGEMENT = "risk_management"
    PHARMACY_DIRECTOR = "pharmacy_director"
    UNIT_MANAGER = "unit_manager"
    PATIENT_RELATIONS = "patient_relations"
    BIOMEDICAL_ENGINEERING = "biomedical_engineering"
    SAFETY_OFFICER = "safety_officer"
    COMPLIANCE = "compliance"
    LEGAL = "legal"
    DEPARTMENT_HEAD = "department_head"
    INFECTION_CONTROL = "infection_control"
    QUALITY_IMPROVEMENT = "quality_improvement"
    HR_DEPARTMENT = "hr_department"


@dataclass
class ClassificationResult:
    """Result of AI classification"""
    primary_category: IncidentCategory
    secondary_categories: List[IncidentCategory] = field(default_factory=list)
    department: Department = Department.GENERAL
    severity: SeverityLevel = SeverityLevel.MINOR_HARM
    confidence_scores: dict = field(default_factory=dict)
    extracted_entities: dict = field(default_factory=dict)
    sentiment_score: float = 0.0  # -1 (negative) to 1 (positive)
    

@dataclass
class PriorityScore:
    """Computed priority score with breakdown"""
    total_score: float  # 1-10 scale
    severity_component: float
    recurrence_component: float
    patient_impact_component: float
    regulatory_component: float
    urgency_level: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    recommended_sla_hours: int


@dataclass
class RoutingDecision:
    """Routing decision with assignments"""
    primary_assignee: RouteDestination
    secondary_assignees: List[RouteDestination]
    escalation_path: List[RouteDestination]
    requires_immediate_attention: bool
    notification_channels: List[str]  # "email", "sms", "pager", "dashboard"


@dataclass
class IncidentReport:
    """Incoming incident report"""
    id: str
    submitted_at: datetime
    report_text: str
    reporter_role: Optional[str] = None
    patient_id: Optional[str] = None  # De-identified
    location: Optional[str] = None
    
    # Populated after processing
    classification: Optional[ClassificationResult] = None
    priority: Optional[PriorityScore] = None
    routing: Optional[RoutingDecision] = None
    processing_timestamp: Optional[datetime] = None


@dataclass
class TriageResult:
    """Complete triage result for an incident"""
    incident: IncidentReport
    classification: ClassificationResult
    priority: PriorityScore
    routing: RoutingDecision
    similar_incidents: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    requires_human_review: bool = False
    review_reason: Optional[str] = None
