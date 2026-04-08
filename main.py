"""
Patient Complaint & Incident Report Triage System

AI-powered triage using Meta Llama 3.3 70B via HuggingFace (FREE!)

Pipeline:
1. Classification - Categorize incidents by type, department, severity
2. Priority Scoring - Compute urgency and risk scores  
3. Routing - Assign to appropriate personnel with escalation paths

Usage:
    python main.py                    # Run demo with sample incidents
    python main.py --interactive      # Interactive mode for custom input

Requirements:
    - HUGGINGFACE_API_KEY in .env file (get free at huggingface.co/settings/tokens)
    - pip install huggingface_hub
"""

import argparse
import time
from datetime import datetime
from typing import Optional, Dict

from models import IncidentReport, TriageResult, ClassificationResult
from llm_classifier import LLMClassifier, create_llm_classifier
from priority_scorer import PriorityScoringEngine
from router import RoutingEngine, WorkflowEngine
from sample_data import get_all_sample_incidents, generate_historical_incidents


class TriageSystem:
    """
    Complete triage system using Llama 3.3 70B for intelligent classification.
    """
    
    def __init__(self):
        """Initialize the triage system with Llama 3.3."""
        print("Initializing Patient Triage System...")
        print("🤖 Using Meta Llama 3.3 70B via HuggingFace\n")
        
        # Initialize LLM classifier
        self.classifier = create_llm_classifier()
        
        # Load historical data for recurrence detection
        historical_data = generate_historical_incidents(50)
        self.priority_scorer = PriorityScoringEngine(historical_data)
        
        self.workflow = WorkflowEngine()
        
        print("\nSystem initialized successfully!\n")
    
    def process_incident(self, report: IncidentReport) -> tuple:
        """
        Process a single incident through the complete triage pipeline.
        
        Returns:
            Tuple of (TriageResult, llm_analysis dict)
        """
        start_time = time.time()
        
        # Step 1: AI Classification with Llama 3.3
        classification, llm_analysis = self.classifier.classify(report.report_text)
        
        # Step 2: Priority Scoring
        priority = self.priority_scorer.compute_priority(
            classification=classification,
            report_text=report.report_text,
            department=classification.department.value,
            location=report.location
        )
        
        # Step 3: Routing
        routing = self.workflow.process_routing(
            incident_id=report.id,
            classification=classification,
            priority=priority,
            report_summary=report.report_text[:200]
        )
        
        # Determine if human review needed
        needs_review = False
        review_reason = None
        
        confidence = classification.confidence_scores.get("primary_category", 1.0)
        if confidence < 0.7:
            needs_review = True
            review_reason = f"Low classification confidence: {confidence:.1%}"
        
        # Build result
        processing_time = (time.time() - start_time) * 1000
        
        result = TriageResult(
            incident=report,
            classification=classification,
            priority=priority,
            routing=routing,
            similar_incidents=[],
            processing_time_ms=processing_time,
            requires_human_review=needs_review,
            review_reason=review_reason
        )
        
        return result, llm_analysis


def print_triage_result(result: TriageResult, llm_analysis: Optional[Dict] = None):
    """Pretty print a triage result."""
    print("=" * 70)
    print(f"INCIDENT: {result.incident.id}")
    print("=" * 70)
    
    # Report summary
    text_preview = result.incident.report_text.strip()[:200].replace('\n', ' ')
    print(f"\n📋 REPORT SUMMARY:")
    print(f"   {text_preview}...")
    
    # Classification
    print(f"\n🏷️  CLASSIFICATION:")
    print(f"   Category:    {result.classification.primary_category.value.upper()}")
    if result.classification.secondary_categories:
        secondary = [c.value for c in result.classification.secondary_categories]
        print(f"   Secondary:   {', '.join(secondary)}")
    print(f"   Department:  {result.classification.department.value}")
    print(f"   Severity:    {result.classification.severity.name}")
    
    if result.classification.extracted_entities:
        print(f"   Entities:    {result.classification.extracted_entities}")
    
    confidence = result.classification.confidence_scores.get("primary_category", 0)
    print(f"   Confidence:  {confidence:.1%}")
    
    # LLM-specific analysis
    if llm_analysis:
        print(f"\n🤖 AI ANALYSIS (Llama 3.3):")
        if llm_analysis.get("summary"):
            print(f"   Summary:     {llm_analysis['summary']}")
        if llm_analysis.get("reasoning"):
            print(f"   Reasoning:   {llm_analysis['reasoning']}")
        if llm_analysis.get("root_cause_hints"):
            print(f"   Root Causes: {', '.join(llm_analysis['root_cause_hints'][:3])}")
        if llm_analysis.get("immediate_actions"):
            print(f"   Actions:")
            for action in llm_analysis["immediate_actions"][:3]:
                print(f"     • {action}")
    
    # Priority
    print(f"\n⚡ PRIORITY:")
    print(f"   Score:       {result.priority.total_score}/10")
    print(f"   Urgency:     {result.priority.urgency_level}")
    print(f"   SLA:         {result.priority.recommended_sla_hours} hours")
    print(f"   Breakdown:")
    print(f"     • Severity:    {result.priority.severity_component:.2f}")
    print(f"     • Recurrence:  {result.priority.recurrence_component:.2f}")
    print(f"     • Impact:      {result.priority.patient_impact_component:.2f}")
    print(f"     • Regulatory:  {result.priority.regulatory_component:.2f}")
    
    # Routing
    print(f"\n📬 ROUTING:")
    print(f"   Primary:     {result.routing.primary_assignee.value}")
    if result.routing.secondary_assignees:
        secondary = [a.value for a in result.routing.secondary_assignees]
        print(f"   CC:          {', '.join(secondary)}")
    print(f"   Escalation:  {' → '.join(a.value for a in result.routing.escalation_path)}")
    print(f"   Notify via:  {', '.join(result.routing.notification_channels)}")
    print(f"   Immediate:   {'🚨 YES' if result.routing.requires_immediate_attention else 'No'}")
    
    # Human review
    if result.requires_human_review:
        print(f"\n⚠️  REQUIRES HUMAN REVIEW: {result.review_reason}")
    
    print(f"\n⏱️  Processed in {result.processing_time_ms:.1f}ms")
    print()


def run_demo():
    """Run demonstration with sample incidents."""
    print("\n" + "=" * 70)
    print("   PATIENT COMPLAINT & INCIDENT REPORT TRIAGE SYSTEM")
    print("   🤖 AI-Powered with Meta Llama 3.3 70B (HuggingFace FREE)")
    print("=" * 70 + "\n")
    
    # Initialize system
    system = TriageSystem()
    
    # Process sample incidents (limit to 3 for demo)
    samples = get_all_sample_incidents()[:3]
    print(f"Processing {len(samples)} sample incidents...\n")
    
    # Statistics
    urgency_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    category_counts = {}
    total_time = 0
    
    for sample in samples:
        report = IncidentReport(
            id=sample["id"],
            submitted_at=datetime.now(),
            report_text=sample["text"],
            reporter_role=sample.get("reporter_role"),
            location=sample.get("location")
        )
        
        result, llm_analysis = system.process_incident(report)
        print_triage_result(result, llm_analysis)
        
        urgency_counts[result.priority.urgency_level] += 1
        cat = result.classification.primary_category.value
        category_counts[cat] = category_counts.get(cat, 0) + 1
        total_time += result.processing_time_ms
    
    # Print summary
    print("\n" + "=" * 70)
    print("   SUMMARY")
    print("=" * 70)
    print(f"\n📊 Incidents by Urgency:")
    for urgency, count in urgency_counts.items():
        bar = "█" * count
        print(f"   {urgency:10} {bar} ({count})")
    
    print(f"\n📁 Incidents by Category:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"   {cat:25} {count}")
    
    print(f"\n⏱️  Average processing time: {total_time/len(samples):.1f}ms per incident")
    print(f"\n🤖 Model: meta-llama/Llama-3.3-70B-Instruct")
    print()


def run_interactive():
    """Interactive mode for custom incident input."""
    print("\n" + "=" * 70)
    print("   PATIENT TRIAGE SYSTEM - INTERACTIVE MODE")
    print("   🤖 Powered by Meta Llama 3.3 70B")
    print("=" * 70 + "\n")
    
    system = TriageSystem()
    incident_num = 1
    
    while True:
        print("\nEnter incident report (or 'quit' to exit):")
        print("(Press Enter twice when done)")
        print("-" * 40)
        
        lines = []
        while True:
            line = input()
            if line.lower() == 'quit':
                print("\nExiting. Goodbye!")
                return
            if line == '':
                if lines:
                    break
            else:
                lines.append(line)
        
        report_text = '\n'.join(lines)
        
        if not report_text.strip():
            continue
        
        report = IncidentReport(
            id=f"INTERACTIVE-{incident_num:03d}",
            submitted_at=datetime.now(),
            report_text=report_text
        )
        
        result, llm_analysis = system.process_incident(report)
        print_triage_result(result, llm_analysis)
        
        incident_num += 1


def main():
    parser = argparse.ArgumentParser(
        description="Patient Complaint & Incident Report Triage System (Llama 3.3)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode for custom input"
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        run_interactive()
    else:
        run_demo()


if __name__ == "__main__":
    main()
