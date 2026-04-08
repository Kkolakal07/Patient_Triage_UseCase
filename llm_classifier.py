"""
LLM Integration Module for Patient Triage System

Uses HuggingFace Inference API with Meta Llama 3.3 70B Instruct model
for AI-powered healthcare incident classification.

Features:
1. Classification with semantic understanding
2. Severity assessment with clinical reasoning
3. Entity extraction from unstructured text
4. Incident summarization
5. Root cause analysis
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from models import (
    ClassificationResult, IncidentCategory, Department, 
    SeverityLevel, IncidentReport
)


def load_env_file():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if key not in os.environ:
                        os.environ[key] = value

# Load .env on module import
load_env_file()


# ============================================================================
# CONFIGURATION - Llama 3.3 70B via HuggingFace (FREE!)
# ============================================================================
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
TEMPERATURE = 0.1  # Low temp for consistent classification
MAX_TOKENS = 1000


# System prompt for incident classification
CLASSIFICATION_SYSTEM_PROMPT = """You are an expert healthcare incident analyst specializing in patient safety, quality improvement, and health equity. Your role is to analyze patient incident reports and complaints while considering patient demographics and vulnerable populations.

IMPORTANT DEMOGRAPHIC CONSIDERATIONS:
- Age: Pediatric (<18) and geriatric (>65) patients are more vulnerable
- Language barriers: Non-English speakers face communication risks
- Socioeconomic factors: Uninsured/Medicaid patients may face care access issues
- Cultural sensitivity: Consider race/ethnicity in care quality assessment
- Gender-specific care: Some incidents may have gender-related implications

For each incident, you must provide a structured analysis with:

1. PRIMARY_CATEGORY: The main incident type. Choose exactly ONE from:
   - medication_error: Wrong drug, dose, patient, route, or timing
   - patient_fall: Falls, slips, found on floor
   - infection: HAI, surgical site infection, contamination
   - equipment_failure: Device malfunction, alarm failure
   - communication: Handoff errors, miscommunication, language barriers
   - diagnosis_error: Missed, delayed, or incorrect diagnosis
   - surgical_error: Wrong site/side, retained objects, complications
   - patient_experience: Service complaints, wait times, staff attitude
   - delay_in_care: Delayed treatment, slow response
   - privacy_breach: HIPAA violations, unauthorized access
   - staff_conduct: Behavioral issues, harassment, impairment
   - documentation: Missing or incorrect documentation
   - other: Does not fit above categories

2. SECONDARY_CATEGORIES: Up to 2 additional relevant categories (can be empty)

3. DEPARTMENT: Where the incident occurred. Choose ONE from:
   - emergency, surgery, icu, radiology, pharmacy, nursing, 
   - laboratory, administration, outpatient, pediatrics, 
   - maternity, cardiology, oncology, general

4. SEVERITY: Assess the harm level. Choose ONE from:
   - NEAR_MISS: Error occurred but did not reach the patient
   - MINOR_HARM: Temporary harm, no intervention needed
   - MODERATE_HARM: Temporary harm requiring intervention
   - SERIOUS_HARM: Permanent harm or prolonged hospitalization  
   - SENTINEL_EVENT: Death or serious permanent injury (wrong-site surgery, infant abduction, etc.)

5. ENTITIES: Extract key information:
   - medications: Any drug names mentioned
   - staff_roles: nurse, doctor, pharmacist, etc.
   - times: Specific times mentioned
   - locations: Room numbers, units, areas
   - procedures: Any medical procedures mentioned

6. SUMMARY: A 1-2 sentence summary of the incident

7. ROOT_CAUSE_HINTS: Potential contributing factors (2-3 items)

8. IMMEDIATE_ACTIONS: Recommended immediate actions (2-3 items)

9. CONFIDENCE: Your confidence in this classification (0.0-1.0)

10. REASONING: Brief explanation of your classification decision

11. DEMOGRAPHIC_RISK_FACTORS: Identify any demographic-related risks (array of strings):
    - "vulnerable_age" if pediatric or elderly patient
    - "language_barrier" if non-English speaker
    - "socioeconomic_risk" if uninsured/underinsured
    - "health_equity_concern" if potential disparate treatment
    - "cultural_consideration" if cultural factors may affect care

12. PRIORITY_ADJUSTMENT: Recommend priority adjustment based on demographics:
    - "increase" if vulnerable population warrants higher priority
    - "standard" if no demographic adjustment needed
    - Provide brief rationale

Respond ONLY with valid JSON in this exact format:
{
    "primary_category": "category_name",
    "secondary_categories": ["cat1", "cat2"],
    "department": "department_name",
    "severity": "SEVERITY_LEVEL",
    "entities": {
        "medications": [],
        "staff_roles": [],
        "times": [],
        "locations": [],
        "procedures": []
    },
    "summary": "Brief summary",
    "root_cause_hints": ["hint1", "hint2"],
    "immediate_actions": ["action1", "action2"],
    "confidence": 0.95,
    "reasoning": "Explanation",
    "demographic_risk_factors": ["vulnerable_age", "language_barrier"],
    "priority_adjustment": {"recommendation": "increase", "rationale": "Elderly patient with language barrier"}
}"""


class LLMClassifier:
    """
    LLM-powered incident classifier using HuggingFace with Llama 3.3 70B.
    
    This classifier uses Meta's Llama 3.3 70B Instruct model via the
    HuggingFace Inference API (FREE tier) for intelligent healthcare
    incident classification.
    """
    
    def __init__(self):
        self.api_key = None
        self.client = None
        self.model = MODEL
        self._initialize()
    
    def _initialize(self):
        """Initialize the HuggingFace client"""
        self.api_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
        
        if not self.api_key:
            raise ValueError(
                "HuggingFace API key not found. Set HUGGINGFACE_API_KEY in .env file.\n"
                "Get your FREE key at: https://huggingface.co/settings/tokens"
            )
        
        # Initialize huggingface_hub client
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=self.api_key)
        except ImportError:
            raise ImportError("huggingface_hub not installed. Run: pip install huggingface_hub")
        
        print(f"✓ Llama 3.3 70B classifier initialized")
        print(f"  Model: {self.model}")
    
    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        """Generate completion using Llama 3.3 via HuggingFace"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.client.chat_completion(
            messages=messages,
            model=self.model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        
        return response.choices[0].message.content
    
    def classify(self, report_text: str, demographics: Dict = None) -> Tuple[ClassificationResult, Dict]:
        """
        Classify incident using Llama 3.3 70B.
        
        Args:
            report_text: The incident report text to classify
            demographics: Optional patient demographics dict
            
        Returns:
            Tuple of (ClassificationResult, raw LLM response dict)
        """
        # Build demographics context if provided
        demographics_context = ""
        if demographics:
            demographics_context = f"""

PATIENT DEMOGRAPHICS:
- Age: {demographics.get('age', 'Unknown')}
- Gender: {demographics.get('gender', 'Unknown')}
- Race: {demographics.get('race', 'Unknown')}
- Ethnicity: {demographics.get('ethnicity', 'Unknown')}
- Primary Language: {demographics.get('language', 'Unknown')}
- Insurance: {demographics.get('insurance', 'Unknown')}
"""
        
        user_prompt = f"""Analyze this patient incident report:

---
{report_text}
---{demographics_context}

Consider any demographic factors that may affect priority or indicate vulnerable populations.
Provide your analysis as JSON."""
        
        try:
            response_text = self._complete(CLASSIFICATION_SYSTEM_PROMPT, user_prompt)
            
            # Parse JSON response (handle markdown code blocks)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            llm_result = json.loads(response_text.strip())
            
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse LLM response as JSON: {e}")
            llm_result = self._get_fallback_result()
        except Exception as e:
            print(f"Warning: LLM classification failed: {e}")
            llm_result = self._get_fallback_result()
        
        classification = self._parse_llm_result(llm_result, report_text)
        return classification, llm_result
    
    def _parse_llm_result(self, llm_result: Dict, report_text: str = "") -> ClassificationResult:
        """Convert LLM JSON response to ClassificationResult"""
        
        # Parse primary category
        try:
            primary_cat = IncidentCategory(llm_result.get("primary_category", "other"))
        except ValueError:
            primary_cat = IncidentCategory.OTHER
        
        # Parse secondary categories
        secondary_cats = []
        for cat in llm_result.get("secondary_categories", []):
            try:
                secondary_cats.append(IncidentCategory(cat))
            except ValueError:
                continue
        
        # Parse department
        try:
            department = Department(llm_result.get("department", "general"))
        except ValueError:
            department = Department.GENERAL
        
        # Parse severity
        severity_map = {
            "NEAR_MISS": SeverityLevel.NEAR_MISS,
            "MINOR_HARM": SeverityLevel.MINOR_HARM,
            "MODERATE_HARM": SeverityLevel.MODERATE_HARM,
            "SERIOUS_HARM": SeverityLevel.SERIOUS_HARM,
            "SENTINEL_EVENT": SeverityLevel.SENTINEL_EVENT,
        }
        severity = severity_map.get(
            llm_result.get("severity", "NEAR_MISS"),
            SeverityLevel.NEAR_MISS
        )
        
        # Calculate calibrated confidence scores
        llm_confidence = llm_result.get("confidence", 0.8)
        confidence_scores = self._calculate_calibrated_confidence(
            llm_confidence=llm_confidence,
            llm_result=llm_result,
            report_text=report_text,
            primary_category=primary_cat,
            secondary_categories=secondary_cats
        )
        
        return ClassificationResult(
            primary_category=primary_cat,
            secondary_categories=secondary_cats,
            department=department,
            severity=severity,
            confidence_scores=confidence_scores,
            extracted_entities=llm_result.get("entities", {}),
            sentiment_score=0.0
        )
    
    def _calculate_calibrated_confidence(
        self,
        llm_confidence: float,
        llm_result: Dict,
        report_text: str,
        primary_category: IncidentCategory,
        secondary_categories: List[IncidentCategory]
    ) -> Dict[str, float]:
        """
        Calculate calibrated confidence scores using multiple factors.
        
        This provides more reliable confidence than LLM self-reporting alone by
        combining the LLM's assessment with objective text quality metrics.
        
        Factors considered:
        1. LLM self-reported confidence (baseline)
        2. Text quality (length, detail level)
        3. Entity extraction success (found entities = clearer text)
        4. Category clarity (single clear category vs ambiguous)
        5. Reasoning quality (LLM provided reasoning = more thought)
        
        Returns:
            Dict with calibrated confidence scores for each classification dimension
        """
        
        # === FACTOR 1: Text Quality Score (0.0 - 1.0) ===
        text_length = len(report_text.strip())
        if text_length < 50:
            text_quality = 0.3  # Very short, likely missing context
        elif text_length < 100:
            text_quality = 0.5  # Short
        elif text_length < 200:
            text_quality = 0.7  # Moderate
        elif text_length < 500:
            text_quality = 0.9  # Good detail
        else:
            text_quality = 1.0  # Excellent detail
        
        # === FACTOR 2: Entity Extraction Score (0.0 - 1.0) ===
        entities = llm_result.get("entities", {})
        entity_count = sum(
            len(v) if isinstance(v, list) else (1 if v else 0)
            for v in entities.values()
        )
        if entity_count == 0:
            entity_score = 0.4  # No entities found - vague text
        elif entity_count <= 2:
            entity_score = 0.6  # Few entities
        elif entity_count <= 5:
            entity_score = 0.8  # Good entity extraction
        else:
            entity_score = 1.0  # Rich entity extraction
        
        # === FACTOR 3: Category Clarity Score (0.0 - 1.0) ===
        # Clear single category = high confidence
        # Multiple secondary categories = more ambiguous
        if primary_category == IncidentCategory.OTHER:
            category_clarity = 0.4  # "Other" suggests unclear classification
        elif len(secondary_categories) == 0:
            category_clarity = 1.0  # Clear single category
        elif len(secondary_categories) == 1:
            category_clarity = 0.85  # Mostly clear with minor overlap
        else:
            category_clarity = 0.7  # Multiple overlapping categories
        
        # === FACTOR 4: Reasoning Quality (0.0 - 1.0) ===
        reasoning = llm_result.get("reasoning", "")
        if len(reasoning) < 20:
            reasoning_score = 0.5  # Minimal or no reasoning
        elif len(reasoning) < 50:
            reasoning_score = 0.7  # Brief reasoning
        elif len(reasoning) < 100:
            reasoning_score = 0.85  # Good reasoning
        else:
            reasoning_score = 1.0  # Detailed reasoning
        
        # === FACTOR 5: Severity Indicator Presence (0.0 - 1.0) ===
        severity = llm_result.get("severity", "NEAR_MISS")
        severity_keywords = {
            "death", "died", "fatal", "permanent", "serious", "critical",
            "wrong site", "wrong patient", "sentinel", "harm", "injury",
            "fall", "medication error", "infection"
        }
        text_lower = report_text.lower()
        keyword_matches = sum(1 for kw in severity_keywords if kw in text_lower)
        
        # High severity should have keyword evidence
        if severity in ["SERIOUS_HARM", "SENTINEL_EVENT"]:
            severity_score = min(0.5 + (keyword_matches * 0.1), 1.0)
        else:
            severity_score = 0.9  # Lower severities don't need strong evidence
        
        # === COMBINE FACTORS (Weighted Average) ===
        # LLM confidence is the base, but we adjust based on objective factors
        weights = {
            "llm": 0.30,           # LLM self-assessment
            "text_quality": 0.20,  # How detailed is the report?
            "entity": 0.15,        # Did we extract meaningful entities?
            "category": 0.20,      # Is classification clear?
            "reasoning": 0.10,     # Did LLM explain its decision?
            "severity": 0.05,      # Does severity match text evidence?
        }
        
        # Primary category confidence (all factors)
        primary_confidence = (
            weights["llm"] * llm_confidence +
            weights["text_quality"] * text_quality +
            weights["entity"] * entity_score +
            weights["category"] * category_clarity +
            weights["reasoning"] * reasoning_score +
            weights["severity"] * severity_score
        )
        
        # Department confidence (similar but less weight on category clarity)
        dept_confidence = (
            weights["llm"] * llm_confidence +
            weights["text_quality"] * text_quality +
            weights["entity"] * entity_score * 1.2 +  # Entities help identify department
            0.10 * category_clarity +
            weights["reasoning"] * reasoning_score
        )
        dept_confidence = min(dept_confidence, 1.0)  # Cap at 1.0
        
        # Severity confidence (weight severity keywords more)
        severity_confidence = (
            weights["llm"] * llm_confidence +
            weights["text_quality"] * text_quality +
            0.25 * severity_score +  # Severity keywords matter most
            weights["reasoning"] * reasoning_score
        )
        severity_confidence = min(severity_confidence, 1.0)
        
        return {
            "primary_category": round(primary_confidence, 3),
            "department": round(dept_confidence, 3),
            "severity": round(severity_confidence, 3),
            # Include component scores for transparency
            "_components": {
                "llm_self_reported": round(llm_confidence, 3),
                "text_quality": round(text_quality, 3),
                "entity_extraction": round(entity_score, 3),
                "category_clarity": round(category_clarity, 3),
                "reasoning_quality": round(reasoning_score, 3),
                "severity_evidence": round(severity_score, 3),
            }
        }
    
    def _get_fallback_result(self) -> Dict:
        """Return fallback result when LLM fails"""
        return {
            "primary_category": "other",
            "secondary_categories": [],
            "department": "general",
            "severity": "NEAR_MISS",
            "entities": {},
            "summary": "Classification failed - manual review required",
            "root_cause_hints": [],
            "immediate_actions": ["Manual review required"],
            "confidence": 0.0,
            "reasoning": "LLM classification failed"
        }
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for similarity search"""
        embedding = self.client.feature_extraction(
            text, 
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)


def create_llm_classifier(**kwargs) -> LLMClassifier:
    """
    Factory function to create the LLM classifier.
    
    Uses HuggingFace with Llama 3.3 70B Instruct (FREE!)
    
    Returns:
        Configured LLMClassifier instance
    
    Example:
        classifier = create_llm_classifier()
        result, analysis = classifier.classify("Patient fell...")
    """
    return LLMClassifier()


if __name__ == "__main__":
    # Quick test
    sample = """
    Patient in ICU received double dose of morphine due to confusion during 
    shift change. Night nurse had already administered 4mg morphine at 2:00 AM, 
    but this was not properly documented. Patient became over-sedated and 
    required Narcan. Patient recovered fully but family is upset.
    """
    
    print("Testing LLM Classifier with Llama 3.3 70B...")
    print("=" * 60)
    
    try:
        classifier = create_llm_classifier()
        classification, result = classifier.classify(sample)
        
        print(f"\n🏷️  Category: {classification.primary_category.value}")
        print(f"⚠️  Severity: {classification.severity.name}")
        print(f"🏥 Department: {classification.department.value}")
        print(f"📊 Confidence: {classification.confidence_scores.get('primary_category', 0):.0%}")
        print(f"\n📝 Summary: {result.get('summary', 'N/A')}")
        print(f"🔍 Root Causes: {result.get('root_cause_hints', [])}")
        
    except Exception as e:
        print(f"\n⚠️  Test failed: {e}")
        print("\nMake sure HUGGINGFACE_API_KEY is set in your .env file")
