"""
Text Classification Module for Patient Incident Reports

Uses NLP/ML to classify incidents by category, department, and severity.
Can use transformer models (ClinicalBERT) or fall back to pattern matching.
"""

import re
from typing import List, Tuple, Dict
from collections import defaultdict

from models import (
    ClassificationResult, IncidentCategory, Department, SeverityLevel
)
from config import (
    CATEGORY_PATTERNS, DEPARTMENT_KEYWORDS, SEVERITY_KEYWORDS, 
    CLASSIFICATION_CONFIG
)


class IncidentClassifier:
    """
    Classifies patient incident reports using NLP.
    
    Supports:
    - Pattern-based classification (default, no dependencies)
    - Transformer-based classification (requires transformers library)
    """
    
    def __init__(self, use_transformer: bool = False):
        self.use_transformer = use_transformer
        self.transformer_model = None
        self.tokenizer = None
        
        if use_transformer:
            self._load_transformer_model()
    
    def _load_transformer_model(self):
        """Load transformer model for advanced classification"""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            model_name = CLASSIFICATION_CONFIG["model_name"]
            print(f"Loading transformer model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transformer_model = AutoModel.from_pretrained(model_name)
            
            if CLASSIFICATION_CONFIG["use_gpu"] and torch.cuda.is_available():
                self.transformer_model = self.transformer_model.cuda()
            
            print("Transformer model loaded successfully")
            
        except ImportError:
            print("Warning: transformers library not available. Using pattern-based classification.")
            self.use_transformer = False
        except Exception as e:
            print(f"Warning: Failed to load transformer model: {e}. Using pattern-based classification.")
            self.use_transformer = False
    
    def classify(self, text: str) -> ClassificationResult:
        """
        Main classification method - analyzes text and returns classification result.
        """
        # Preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Classify by category
        categories, category_scores = self._classify_categories(cleaned_text)
        
        # Detect department
        department, dept_confidence = self._detect_department(cleaned_text)
        
        # Assess severity
        severity, severity_confidence = self._assess_severity(cleaned_text)
        
        # Extract entities
        entities = self._extract_entities(cleaned_text)
        
        # Compute sentiment (simplified)
        sentiment = self._analyze_sentiment(cleaned_text)
        
        # Build confidence scores
        confidence_scores = {
            "primary_category": category_scores.get(categories[0].value, 0) if categories else 0,
            "department": dept_confidence,
            "severity": severity_confidence,
        }
        
        return ClassificationResult(
            primary_category=categories[0] if categories else IncidentCategory.OTHER,
            secondary_categories=categories[1:3] if len(categories) > 1 else [],
            department=department,
            severity=severity,
            confidence_scores=confidence_scores,
            extracted_entities=entities,
            sentiment_score=sentiment
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Expand common medical abbreviations
        abbreviations = {
            r'\bpt\b': 'patient',
            r'\bpts\b': 'patients',
            r'\bmd\b': 'doctor',
            r'\brn\b': 'nurse',
            r'\bmeds\b': 'medications',
            r'\bmed\b': 'medication',
            r'\badm\b': 'administration',
            r'\bw/\b': 'with',
            r'\bw/o\b': 'without',
            r'\bb/p\b': 'blood pressure',
            r'\bhr\b': 'heart rate',
            r'\bpo\b': 'by mouth',
            r'\biv\b': 'intravenous',
            r'\bim\b': 'intramuscular',
        }
        
        for abbrev, expansion in abbreviations.items():
            text = re.sub(abbrev, expansion, text)
        
        return text
    
    def _classify_categories(self, text: str) -> Tuple[List[IncidentCategory], Dict[str, float]]:
        """
        Classify text into incident categories.
        Returns sorted list of categories and their scores.
        """
        scores = defaultdict(float)
        
        for category, patterns in CATEGORY_PATTERNS.items():
            for pattern in patterns:
                # Count occurrences and weight by pattern specificity
                matches = len(re.findall(r'\b' + re.escape(pattern) + r'\b', text, re.IGNORECASE))
                if matches > 0:
                    # More specific (longer) patterns get higher weight
                    weight = len(pattern.split()) * 0.5 + 1
                    scores[category] += matches * weight
        
        # Normalize scores to 0-1 range
        max_score = max(scores.values()) if scores else 1
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}
        
        # Sort by score and convert to enum
        sorted_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        categories = []
        for cat_name, score in sorted_categories:
            if score > 0.1:  # Minimum threshold
                try:
                    categories.append(IncidentCategory(cat_name))
                except ValueError:
                    continue
        
        if not categories:
            categories = [IncidentCategory.OTHER]
            scores["other"] = 0.5
        
        return categories, dict(scores)
    
    def _detect_department(self, text: str) -> Tuple[Department, float]:
        """Detect the most likely department from text"""
        scores = defaultdict(float)
        
        for dept, keywords in DEPARTMENT_KEYWORDS.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                    scores[dept] += 1
        
        if not scores:
            return Department.GENERAL, 0.5
        
        # Get highest scoring department
        best_dept = max(scores.items(), key=lambda x: x[1])
        max_possible = max(len(kw) for kw in DEPARTMENT_KEYWORDS.values())
        confidence = min(best_dept[1] / 3, 1.0)  # Normalize
        
        try:
            return Department(best_dept[0]), confidence
        except ValueError:
            return Department.GENERAL, 0.5
    
    def _assess_severity(self, text: str) -> Tuple[SeverityLevel, float]:
        """
        Assess incident severity based on keyword detection.
        Uses a conservative approach - always flags potential high-severity events.
        """
        severity_score = 0
        matched_keywords = []
        
        for level, config in SEVERITY_KEYWORDS.items():
            for keyword in config["keywords"]:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                    severity_score = max(severity_score, config["severity_boost"])
                    matched_keywords.append(keyword)
        
        # Map score to severity level
        if severity_score >= 5:
            severity = SeverityLevel.SENTINEL_EVENT
            confidence = 0.95
        elif severity_score >= 4:
            severity = SeverityLevel.SERIOUS_HARM
            confidence = 0.85
        elif severity_score >= 3:
            severity = SeverityLevel.MODERATE_HARM
            confidence = 0.75
        elif severity_score >= 1:
            severity = SeverityLevel.MINOR_HARM
            confidence = 0.70
        else:
            severity = SeverityLevel.NEAR_MISS
            confidence = 0.60
        
        return severity, confidence
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract relevant entities from text"""
        entities = {
            "medications": [],
            "procedures": [],
            "body_parts": [],
            "staff_roles": [],
            "times": [],
            "locations": [],
        }
        
        # Common medication patterns
        med_patterns = [
            r'\b(aspirin|tylenol|morphine|insulin|heparin|warfarin|metformin|lisinopril|'
            r'amoxicillin|prednisone|omeprazole|metoprolol|atorvastatin|amlodipine|'
            r'gabapentin|hydrocodone|oxycodone|fentanyl|vancomycin|ceftriaxone)\b'
        ]
        for pattern in med_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["medications"].extend([m.lower() for m in matches])
        
        # Staff roles
        role_pattern = r'\b(nurse|doctor|physician|surgeon|pharmacist|technician|resident|attending|np|pa|cna|rn|md)\b'
        matches = re.findall(role_pattern, text, re.IGNORECASE)
        entities["staff_roles"] = list(set([m.lower() for m in matches]))
        
        # Time references
        time_pattern = r'\b(\d{1,2}:\d{2}\s*(?:am|pm)?|\d{1,2}\s*(?:am|pm)|morning|afternoon|evening|night|midnight|noon)\b'
        matches = re.findall(time_pattern, text, re.IGNORECASE)
        entities["times"] = list(set(matches))
        
        # Location references
        location_pattern = r'\b(room\s*\d+|bed\s*\d+|unit\s*\d+|floor\s*\d+|hall(?:way)?|bathroom|lobby)\b'
        matches = re.findall(location_pattern, text, re.IGNORECASE)
        entities["locations"] = list(set(matches))
        
        # Clean up empty lists
        entities = {k: v for k, v in entities.items() if v}
        
        return entities
    
    def _analyze_sentiment(self, text: str) -> float:
        """
        Simple sentiment analysis for patient distress level.
        Returns -1 (very negative) to 1 (positive)
        """
        negative_words = [
            'angry', 'frustrated', 'upset', 'furious', 'terrible', 'horrible',
            'worst', 'unacceptable', 'negligent', 'incompetent', 'disgusted',
            'traumatic', 'painful', 'suffering', 'agony', 'scared', 'frightened',
            'abandoned', 'ignored', 'mistreated', 'abused'
        ]
        
        positive_words = [
            'thankful', 'grateful', 'appreciate', 'excellent', 'wonderful',
            'caring', 'professional', 'helpful', 'resolved', 'satisfied'
        ]
        
        neg_count = sum(1 for word in negative_words if word in text)
        pos_count = sum(1 for word in positive_words if word in text)
        
        total = neg_count + pos_count
        if total == 0:
            return 0.0
        
        # Calculate sentiment score
        sentiment = (pos_count - neg_count) / total
        return round(sentiment, 2)


class TransformerClassifier(IncidentClassifier):
    """
    Advanced classifier using transformer embeddings for semantic similarity.
    Requires: transformers, torch, scikit-learn
    """
    
    def __init__(self):
        super().__init__(use_transformer=True)
        self.category_embeddings = {}
        
        if self.transformer_model is not None:
            self._build_category_embeddings()
    
    def _build_category_embeddings(self):
        """Pre-compute embeddings for category descriptions"""
        category_descriptions = {
            IncidentCategory.MEDICATION_ERROR: "medication drug prescription dosing pharmacy error adverse reaction",
            IncidentCategory.PATIENT_FALL: "patient fell fall injury slip trip floor bed wheelchair",
            IncidentCategory.INFECTION: "infection sepsis contamination bacteria virus HAI MRSA",
            IncidentCategory.EQUIPMENT_FAILURE: "equipment device malfunction broken failure monitor pump",
            IncidentCategory.COMMUNICATION: "communication miscommunication handoff language barrier consent",
            IncidentCategory.DIAGNOSIS_ERROR: "diagnosis misdiagnosis delayed missed test results imaging",
            IncidentCategory.SURGICAL_ERROR: "surgery surgical wrong site retained object complication",
            IncidentCategory.PATIENT_EXPERIENCE: "complaint rude unprofessional wait time dissatisfied billing",
            IncidentCategory.DELAY_IN_CARE: "delay waiting slow response neglected unattended",
            IncidentCategory.PRIVACY_BREACH: "privacy HIPAA confidential records breach unauthorized",
            IncidentCategory.STAFF_CONDUCT: "behavior conduct harassment discrimination abuse attitude",
        }
        
        for category, description in category_descriptions.items():
            embedding = self._get_embedding(description)
            if embedding is not None:
                self.category_embeddings[category] = embedding
    
    def _get_embedding(self, text: str):
        """Get transformer embedding for text"""
        if self.transformer_model is None:
            return None
        
        try:
            import torch
            
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=CLASSIFICATION_CONFIG["max_text_length"],
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.transformer_model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].squeeze()
            
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def _classify_with_embeddings(self, text: str) -> Tuple[List[IncidentCategory], Dict[str, float]]:
        """Classify using semantic similarity to category embeddings"""
        text_embedding = self._get_embedding(text)
        
        if text_embedding is None or not self.category_embeddings:
            # Fall back to pattern matching
            return super()._classify_categories(text)
        
        import torch
        from torch.nn.functional import cosine_similarity
        
        scores = {}
        for category, cat_embedding in self.category_embeddings.items():
            similarity = cosine_similarity(
                text_embedding.unsqueeze(0), 
                cat_embedding.unsqueeze(0)
            ).item()
            scores[category.value] = (similarity + 1) / 2  # Normalize to 0-1
        
        # Sort and filter
        sorted_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        categories = [IncidentCategory(cat) for cat, score in sorted_categories if score > 0.3]
        
        if not categories:
            categories = [IncidentCategory.OTHER]
        
        return categories, scores


def create_classifier(use_advanced: bool = False) -> IncidentClassifier:
    """Factory function to create appropriate classifier"""
    if use_advanced:
        try:
            return TransformerClassifier()
        except Exception as e:
            print(f"Could not create transformer classifier: {e}")
            return IncidentClassifier(use_transformer=False)
    return IncidentClassifier(use_transformer=False)
