"""
REST API for the Patient Triage System

Provides HTTP endpoints for integrating the triage system with hospital
information systems, EHR platforms, and incident reporting portals.

Run with: uvicorn api:app --reload
"""

from datetime import datetime
from typing import List, Optional
import json

# Note: FastAPI is optional. This file serves as a template.
# Install with: pip install fastapi uvicorn

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Install with: pip install fastapi uvicorn")

from models import IncidentReport, TriageResult
from classifier import create_classifier
from priority_scorer import PriorityScoringEngine
from router import WorkflowEngine
from sample_data import generate_historical_incidents


if FASTAPI_AVAILABLE:
    
    # Pydantic models for API
    class IncidentInput(BaseModel):
        """Input model for incident submission"""
        report_text: str = Field(..., min_length=10, description="Incident report text")
        reporter_role: Optional[str] = Field(None, description="Role of reporter")
        patient_id: Optional[str] = Field(None, description="De-identified patient ID")
        location: Optional[str] = Field(None, description="Location of incident")
        
        class Config:
            json_schema_extra = {
                "example": {
                    "report_text": "Patient received wrong medication...",
                    "reporter_role": "RN",
                    "location": "Unit 3"
                }
            }
    
    
    class ClassificationOutput(BaseModel):
        """Classification result output"""
        primary_category: str
        secondary_categories: List[str]
        department: str
        severity: str
        confidence: float
        extracted_entities: dict
    
    
    class PriorityOutput(BaseModel):
        """Priority scoring output"""
        total_score: float
        urgency_level: str
        sla_hours: int
        severity_component: float
        recurrence_component: float
        impact_component: float
        regulatory_component: float
    
    
    class RoutingOutput(BaseModel):
        """Routing decision output"""
        primary_assignee: str
        secondary_assignees: List[str]
        escalation_path: List[str]
        immediate_attention: bool
        notification_channels: List[str]
    
    
    class TriageOutput(BaseModel):
        """Complete triage result output"""
        incident_id: str
        submitted_at: str
        processed_at: str
        classification: ClassificationOutput
        priority: PriorityOutput
        routing: RoutingOutput
        requires_human_review: bool
        review_reason: Optional[str]
        processing_time_ms: float
    
    
    class BatchInput(BaseModel):
        """Batch processing input"""
        incidents: List[IncidentInput]
    
    
    class BatchOutput(BaseModel):
        """Batch processing output"""
        results: List[TriageOutput]
        total_processed: int
        total_time_ms: float
    
    
    # Initialize FastAPI app
    app = FastAPI(
        title="Patient Triage System API",
        description="AI-powered classification, prioritization, and routing of patient incidents",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware for web integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize triage components (global instances)
    classifier = create_classifier(use_advanced=False)
    historical_data = generate_historical_incidents(50)
    priority_scorer = PriorityScoringEngine(historical_data)
    workflow_engine = WorkflowEngine()
    
    # Track processed incidents
    incident_store = {}
    incident_counter = 0
    
    
    def process_incident_internal(input_data: IncidentInput) -> TriageOutput:
        """Internal processing function"""
        global incident_counter
        incident_counter += 1
        
        import time
        start_time = time.time()
        
        # Create incident report
        incident_id = f"API-{datetime.now().strftime('%Y%m%d')}-{incident_counter:04d}"
        
        report = IncidentReport(
            id=incident_id,
            submitted_at=datetime.now(),
            report_text=input_data.report_text,
            reporter_role=input_data.reporter_role,
            patient_id=input_data.patient_id,
            location=input_data.location
        )
        
        # Classification
        classification = classifier.classify(report.report_text)
        
        # Priority scoring
        priority = priority_scorer.compute_priority(
            classification=classification,
            report_text=report.report_text,
            department=classification.department.value,
            location=report.location
        )
        
        # Routing
        routing = workflow_engine.process_routing(
            incident_id=incident_id,
            classification=classification,
            priority=priority,
            report_summary=report.report_text[:200]
        )
        
        # Determine review needs
        confidence = classification.confidence_scores.get("primary_category", 1.0)
        needs_review = confidence < 0.7
        review_reason = f"Low confidence: {confidence:.1%}" if needs_review else None
        
        processing_time = (time.time() - start_time) * 1000
        
        # Build output
        result = TriageOutput(
            incident_id=incident_id,
            submitted_at=report.submitted_at.isoformat(),
            processed_at=datetime.now().isoformat(),
            classification=ClassificationOutput(
                primary_category=classification.primary_category.value,
                secondary_categories=[c.value for c in classification.secondary_categories],
                department=classification.department.value,
                severity=classification.severity.name,
                confidence=confidence,
                extracted_entities=classification.extracted_entities
            ),
            priority=PriorityOutput(
                total_score=priority.total_score,
                urgency_level=priority.urgency_level,
                sla_hours=priority.recommended_sla_hours,
                severity_component=priority.severity_component,
                recurrence_component=priority.recurrence_component,
                impact_component=priority.patient_impact_component,
                regulatory_component=priority.regulatory_component
            ),
            routing=RoutingOutput(
                primary_assignee=routing.primary_assignee.value,
                secondary_assignees=[a.value for a in routing.secondary_assignees],
                escalation_path=[a.value for a in routing.escalation_path],
                immediate_attention=routing.requires_immediate_attention,
                notification_channels=routing.notification_channels
            ),
            requires_human_review=needs_review,
            review_reason=review_reason,
            processing_time_ms=processing_time
        )
        
        # Store for retrieval
        incident_store[incident_id] = result
        
        return result
    
    
    # API Endpoints
    
    @app.get("/")
    async def root():
        """API health check"""
        return {
            "service": "Patient Triage System",
            "status": "healthy",
            "version": "1.0.0"
        }
    
    
    @app.post("/api/v1/triage", response_model=TriageOutput)
    async def triage_incident(incident: IncidentInput):
        """
        Process a single incident report through the triage pipeline.
        
        Returns classification, priority score, and routing decision.
        """
        try:
            return process_incident_internal(incident)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.post("/api/v1/triage/batch", response_model=BatchOutput)
    async def triage_batch(batch: BatchInput):
        """
        Process multiple incidents in a single request.
        
        Useful for bulk import from legacy systems.
        """
        import time
        start_time = time.time()
        
        results = []
        for incident in batch.incidents:
            try:
                result = process_incident_internal(incident)
                results.append(result)
            except Exception as e:
                # Log error but continue processing
                print(f"Error processing incident: {e}")
        
        total_time = (time.time() - start_time) * 1000
        
        return BatchOutput(
            results=results,
            total_processed=len(results),
            total_time_ms=total_time
        )
    
    
    @app.get("/api/v1/incidents/{incident_id}", response_model=TriageOutput)
    async def get_incident(incident_id: str):
        """
        Retrieve a previously processed incident by ID.
        """
        if incident_id not in incident_store:
            raise HTTPException(status_code=404, detail="Incident not found")
        return incident_store[incident_id]
    
    
    @app.get("/api/v1/stats")
    async def get_stats():
        """
        Get system statistics and metrics.
        """
        if not incident_store:
            return {
                "total_processed": 0,
                "by_urgency": {},
                "by_category": {}
            }
        
        urgency_counts = {}
        category_counts = {}
        
        for incident in incident_store.values():
            urgency = incident.priority.urgency_level
            urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
            
            category = incident.classification.primary_category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            "total_processed": len(incident_store),
            "by_urgency": urgency_counts,
            "by_category": category_counts
        }
    
    
    @app.post("/api/v1/classify")
    async def classify_only(incident: IncidentInput):
        """
        Perform only classification without full triage.
        
        Useful for testing or when only category is needed.
        """
        classification = classifier.classify(incident.report_text)
        
        return {
            "primary_category": classification.primary_category.value,
            "secondary_categories": [c.value for c in classification.secondary_categories],
            "department": classification.department.value,
            "severity": classification.severity.name,
            "confidence": classification.confidence_scores,
            "entities": classification.extracted_entities,
            "sentiment": classification.sentiment_score
        }


else:
    # Stub for when FastAPI is not available
    def run_api_stub():
        print("=" * 60)
        print("FastAPI is not installed.")
        print("To run the API server, install dependencies:")
        print("  pip install fastapi uvicorn")
        print("Then run:")
        print("  uvicorn api:app --reload")
        print("=" * 60)


if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        run_api_stub()
