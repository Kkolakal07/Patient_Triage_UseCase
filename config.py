"""
Configuration settings for the Patient Triage System

Uses Meta Llama 3.3 70B via HuggingFace Inference API (FREE!)
"""

# ============================================================================
# LLM CONFIGURATION - Llama 3.3 70B via HuggingFace
# ============================================================================
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
LLM_TEMPERATURE = 0.1  # Low temp for consistent classification
LLM_MAX_TOKENS = 1000

# ============================================================================
# PRIORITY SCORING WEIGHTS
# ============================================================================
PRIORITY_WEIGHTS = {
    "severity": 0.40,
    "recurrence": 0.20,
    "patient_impact": 0.20,
    "regulatory": 0.20,
}

# SLA definitions (in hours)
SLA_BY_URGENCY = {
    "CRITICAL": 1,      # Sentinel events - immediate
    "HIGH": 4,          # Same business day
    "MEDIUM": 24,       # Within 24 hours
    "LOW": 72,          # Within 3 business days
}

# ============================================================================
# SEVERITY DETECTION KEYWORDS (used by priority scorer)
# ============================================================================
SEVERITY_KEYWORDS = {
    "sentinel": {
        "keywords": ["death", "died", "deceased", "fatal", "wrong-site", "wrong patient", 
                     "suicide", "homicide", "infant abduction", "rape", "assault"],
        "severity_boost": 5
    },
    "serious": {
        "keywords": ["permanent", "disability", "amputation", "brain damage", "coma",
                     "ventilator", "code blue", "resuscitation", "life-threatening"],
        "severity_boost": 4
    },
    "moderate": {
        "keywords": ["hospitalization", "surgery required", "blood transfusion", 
                     "infection", "sepsis", "fracture", "laceration"],
        "severity_boost": 3
    },
    "minor": {
        "keywords": ["bruise", "minor injury", "discomfort", "rash", "nausea"],
        "severity_boost": 1
    }
}

# ============================================================================
# DEPARTMENT DETECTION KEYWORDS
# ============================================================================
DEPARTMENT_KEYWORDS = {
    "emergency": ["ER", "ED", "emergency department", "emergency room", "trauma", "triage"],
    "surgery": ["OR", "operating room", "surgical", "post-op", "pre-op", "anesthesia"],
    "icu": ["ICU", "intensive care", "critical care", "CCU", "MICU", "SICU"],
    "radiology": ["X-ray", "CT", "MRI", "imaging", "radiology", "ultrasound"],
    "pharmacy": ["pharmacy", "pharmacist", "medication", "dispensing", "prescription"],
    "laboratory": ["lab", "laboratory", "blood draw", "specimen", "pathology"],
    "pediatrics": ["pediatric", "child", "infant", "NICU", "newborn"],
    "maternity": ["maternity", "labor", "delivery", "obstetric", "L&D", "postpartum"],
    "cardiology": ["cardiology", "cardiac", "heart", "cath lab", "ECG", "EKG"],
    "oncology": ["oncology", "cancer", "chemotherapy", "radiation therapy"],
}

# ============================================================================
# ROUTING RULES
# ============================================================================
ROUTING_RULES = {
    "sentinel_event": {
        "primary": "chief_medical_officer",
        "secondary": ["risk_management", "legal", "compliance"],
        "notification": ["pager", "sms", "email"],
        "immediate": True
    },
    "medication_error": {
        "primary": "pharmacy_director",
        "secondary": ["unit_manager", "safety_officer"],
        "notification": ["email", "dashboard"],
        "immediate": False
    },
    "infection": {
        "primary": "infection_control",
        "secondary": ["unit_manager", "safety_officer"],
        "notification": ["email", "dashboard"],
        "immediate": False
    },
    "equipment_failure": {
        "primary": "biomedical_engineering",
        "secondary": ["safety_officer", "unit_manager"],
        "notification": ["email", "dashboard"],
        "immediate": False
    },
    "patient_experience": {
        "primary": "patient_relations",
        "secondary": ["department_head"],
        "notification": ["email"],
        "immediate": False
    },
    "privacy_breach": {
        "primary": "compliance",
        "secondary": ["legal", "risk_management"],
        "notification": ["email", "dashboard"],
        "immediate": False
    },
    "staff_conduct": {
        "primary": "hr_department",
        "secondary": ["department_head", "legal"],
        "notification": ["email"],
        "immediate": False
    },
    "default": {
        "primary": "quality_improvement",
        "secondary": ["unit_manager"],
        "notification": ["email", "dashboard"],
        "immediate": False
    }
}

# ============================================================================
# RECURRENCE DETECTION SETTINGS
# ============================================================================
RECURRENCE_LOOKBACK_DAYS = 90
RECURRENCE_SIMILARITY_THRESHOLD = 0.8
