"""
Patient Triage System - Web UI

Streamlit-based interface for the AI-powered patient incident triage system.
Uses Meta Llama 3.3 70B via HuggingFace (FREE!)

Run with: streamlit run app.py
"""

import streamlit as st
import time
from datetime import datetime

from models import IncidentReport, TriageResult
from llm_classifier import LLMClassifier, create_llm_classifier
from priority_scorer import PriorityScoringEngine
from router import WorkflowEngine
from sample_data import get_all_sample_incidents, generate_historical_incidents


# Page config
st.set_page_config(
    page_title="Patient Triage System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .severity-critical { background-color: #DC3545; color: white; padding: 5px 10px; border-radius: 5px; }
    .severity-high { background-color: #FD7E14; color: white; padding: 5px 10px; border-radius: 5px; }
    .severity-medium { background-color: #FFC107; color: black; padding: 5px 10px; border-radius: 5px; }
    .severity-low { background-color: #28A745; color: white; padding: 5px 10px; border-radius: 5px; }
    .stAlert { margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_system():
    """Initialize the triage system (cached)"""
    classifier = create_llm_classifier()
    historical_data = generate_historical_incidents(50)
    priority_scorer = PriorityScoringEngine(historical_data)
    workflow = WorkflowEngine()
    return classifier, priority_scorer, workflow


def process_incident(classifier, priority_scorer, workflow, report_text: str, incident_id: str = None, demographics: dict = None):
    """Process a single incident"""
    if not incident_id:
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    report = IncidentReport(
        id=incident_id,
        submitted_at=datetime.now(),
        report_text=report_text
    )
    
    start_time = time.time()
    
    # Classification (with demographics if provided)
    classification, llm_analysis = classifier.classify(report_text, demographics)
    
    # Priority
    priority = priority_scorer.compute_priority(
        classification=classification,
        report_text=report_text,
        department=classification.department.value,
        location=None
    )
    
    # Routing
    routing = workflow.process_routing(
        incident_id=incident_id,
        classification=classification,
        priority=priority,
        report_summary=report_text[:200]
    )
    
    processing_time = (time.time() - start_time) * 1000
    
    return {
        "incident_id": incident_id,
        "classification": classification,
        "llm_analysis": llm_analysis,
        "priority": priority,
        "routing": routing,
        "processing_time": processing_time,
        "demographics": demographics
    }


def display_results(result):
    """Display triage results"""
    classification = result["classification"]
    llm_analysis = result["llm_analysis"]
    priority = result["priority"]
    routing = result["routing"]
    demographics = result.get("demographics")
    
    # Header with urgency
    urgency_colors = {
        "CRITICAL": "🔴",
        "HIGH": "🟠", 
        "MEDIUM": "🟡",
        "LOW": "🟢"
    }
    urgency_icon = urgency_colors.get(priority.urgency_level, "⚪")
    
    st.markdown(f"### {urgency_icon} Incident: {result['incident_id']}")
    st.caption(f"Processed in {result['processing_time']:.0f}ms")
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Category",
            value=classification.primary_category.value.upper().replace("_", " ")
        )
    
    with col2:
        st.metric(
            label="Severity",
            value=classification.severity.name.replace("_", " ")
        )
    
    with col3:
        st.metric(
            label="Priority Score",
            value=f"{priority.total_score}/10"
        )
    
    with col4:
        st.metric(
            label="Urgency",
            value=priority.urgency_level
        )
    
    # Tabs for detailed info
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🤖 AI Analysis", "📊 Classification", "⚡ Priority", "📬 Routing", "👤 Demographics"])
    
    with tab1:
        if llm_analysis:
            st.markdown("#### Summary")
            st.info(llm_analysis.get("summary", "N/A"))
            
            st.markdown("#### Reasoning")
            st.write(llm_analysis.get("reasoning", "N/A"))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔍 Root Causes")
                for cause in llm_analysis.get("root_cause_hints", []):
                    st.write(f"• {cause}")
            
            with col2:
                st.markdown("#### ✅ Recommended Actions")
                for action in llm_analysis.get("immediate_actions", []):
                    st.write(f"• {action}")
            
            st.markdown("#### 📝 Extracted Entities")
            entities = llm_analysis.get("entities", {})
            if entities:
                entity_cols = st.columns(5)
                labels = ["Medications", "Staff Roles", "Times", "Locations", "Procedures"]
                keys = ["medications", "staff_roles", "times", "locations", "procedures"]
                for i, (label, key) in enumerate(zip(labels, keys)):
                    with entity_cols[i]:
                        vals = entities.get(key, [])
                        st.markdown(f"**{label}**")
                        if vals:
                            for v in vals:
                                st.write(f"• {v}")
                        else:
                            st.write("—")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Primary Category")
            st.success(f"**{classification.primary_category.value.upper()}**")
            
            if classification.secondary_categories:
                st.markdown("#### Secondary Categories")
                for cat in classification.secondary_categories:
                    st.write(f"• {cat.value}")
        
        with col2:
            st.markdown("#### Department")
            st.info(f"**{classification.department.value.upper()}**")
            
            st.markdown("#### Calibrated Confidence")
            confidence = classification.confidence_scores.get("primary_category", 0)
            
            # Color-coded confidence
            if confidence >= 0.85:
                st.progress(confidence)
                st.success(f"✅ {confidence:.0%} (High)")
            elif confidence >= 0.70:
                st.progress(confidence)
                st.info(f"📊 {confidence:.0%} (Good)")
            elif confidence >= 0.50:
                st.progress(confidence)
                st.warning(f"⚠️ {confidence:.0%} (Moderate)")
            else:
                st.progress(confidence)
                st.error(f"❌ {confidence:.0%} (Low - Manual Review)")
            
            # Show confidence breakdown
            components = classification.confidence_scores.get("_components", {})
            if components:
                with st.expander("📐 Confidence Calculation Details"):
                    st.caption("Multi-factor calibrated confidence scoring")
                    st.write(f"• **LLM Self-Reported:** {components.get('llm_self_reported', 0):.0%}")
                    st.write(f"• **Text Quality:** {components.get('text_quality', 0):.0%}")
                    st.write(f"• **Entity Extraction:** {components.get('entity_extraction', 0):.0%}")
                    st.write(f"• **Category Clarity:** {components.get('category_clarity', 0):.0%}")
                    st.write(f"• **Reasoning Quality:** {components.get('reasoning_quality', 0):.0%}")
                    st.write(f"• **Severity Evidence:** {components.get('severity_evidence', 0):.0%}")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Score Breakdown")
            st.write(f"• **Severity:** {priority.severity_component:.2f}")
            st.write(f"• **Recurrence:** {priority.recurrence_component:.2f}")
            st.write(f"• **Patient Impact:** {priority.patient_impact_component:.2f}")
            st.write(f"• **Regulatory:** {priority.regulatory_component:.2f}")
        
        with col2:
            st.markdown("#### Response Requirements")
            st.write(f"**SLA:** {priority.recommended_sla_hours} hours")
            
            if priority.urgency_level == "CRITICAL":
                st.error("🚨 IMMEDIATE ATTENTION REQUIRED")
            elif priority.urgency_level == "HIGH":
                st.warning("⚠️ Same-day response required")
            elif priority.urgency_level == "MEDIUM":
                st.info("📋 Response within 24 hours")
            else:
                st.success("✅ Standard processing (72 hours)")
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Primary Assignee")
            st.success(f"**{routing.primary_assignee.value.upper().replace('_', ' ')}**")
            
            if routing.secondary_assignees:
                st.markdown("#### CC'd Recipients")
                for assignee in routing.secondary_assignees:
                    st.write(f"• {assignee.value.replace('_', ' ').title()}")
        
        with col2:
            st.markdown("#### Escalation Path")
            path = " → ".join([a.value.replace("_", " ").title() for a in routing.escalation_path])
            st.write(path)
            
            st.markdown("#### Notification Channels")
            for channel in routing.notification_channels:
                emoji = {"pager": "📟", "sms": "📱", "email": "📧", "dashboard": "💻"}.get(channel, "📢")
                st.write(f"{emoji} {channel.upper()}")
            
            if routing.requires_immediate_attention:
                st.error("🚨 REQUIRES IMMEDIATE ATTENTION")
    
    with tab5:
        if demographics:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Patient Information")
                st.write(f"**Age:** {demographics.get('age', 'Unknown')}")
                st.write(f"**Gender:** {demographics.get('gender', 'Unknown')}")
                st.write(f"**Race:** {demographics.get('race', 'Unknown')}")
                st.write(f"**Ethnicity:** {demographics.get('ethnicity', 'Unknown')}")
            
            with col2:
                st.markdown("#### Care Factors")
                st.write(f"**Primary Language:** {demographics.get('language', 'Unknown')}")
                st.write(f"**Insurance:** {demographics.get('insurance', 'Unknown')}")
                
                # Vulnerable population flags
                age = demographics.get('age', 0)
                language = demographics.get('language', 'English')
                insurance = demographics.get('insurance', '')
                
                flags = []
                if age < 18:
                    flags.append("👶 Pediatric Patient")
                elif age >= 65:
                    flags.append("👴 Geriatric Patient")
                if language and language != 'English':
                    flags.append("🌐 Non-English Speaker")
                if insurance in ['Medicaid', 'Uninsured', 'Self-Pay']:
                    flags.append("💳 Socioeconomic Risk")
                
                if flags:
                    st.markdown("#### ⚠️ Vulnerability Flags")
                    for flag in flags:
                        st.warning(flag)
            
            # Show AI-identified demographic risk factors
            if llm_analysis:
                risk_factors = llm_analysis.get("demographic_risk_factors", [])
                priority_adj = llm_analysis.get("priority_adjustment", {})
                
                if risk_factors:
                    st.markdown("---")
                    st.markdown("#### 🔍 AI-Identified Demographic Risk Factors")
                    for factor in risk_factors:
                        factor_labels = {
                            "vulnerable_age": "🎂 Vulnerable Age Group",
                            "language_barrier": "🗣️ Language Barrier Risk",
                            "socioeconomic_risk": "💰 Socioeconomic Risk Factor",
                            "health_equity_concern": "⚖️ Health Equity Concern",
                            "cultural_consideration": "🌍 Cultural Consideration"
                        }
                        st.write(f"• {factor_labels.get(factor, factor)}")
                
                if priority_adj:
                    st.markdown("#### 📈 Priority Adjustment")
                    rec = priority_adj.get('recommendation', 'standard')
                    rationale = priority_adj.get('rationale', 'No adjustment needed')
                    if rec == 'increase':
                        st.error(f"⬆️ **PRIORITY INCREASED:** {rationale}")
                    else:
                        st.info(f"➡️ **Standard Priority:** {rationale}")
        else:
            st.info("No demographic data available for this incident.")
            st.write("Demographics are included with sample incidents from the database.")


def main():
    # Header
    st.markdown('<p class="main-header">🏥 Patient Triage System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Incident Classification using Meta Llama 3.3 70B</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ System Info")
        st.info("**Model:** Llama 3.3 70B\n\n**Provider:** HuggingFace (FREE)")
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.write("""
        This system automatically:
        - 🏷️ **Classifies** incidents by type
        - ⚡ **Prioritizes** based on severity
        - 📬 **Routes** to appropriate staff
        - 🔍 **Analyzes** root causes
        - ✅ **Recommends** actions
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Categories")
        categories = [
            "Medication Error", "Patient Fall", "Infection",
            "Surgical Error", "Equipment Failure", "Communication",
            "Diagnosis Error", "Delay in Care", "Privacy Breach"
        ]
        for cat in categories:
            st.write(f"• {cat}")
    
    # Initialize system
    with st.spinner("🔄 Initializing AI system..."):
        try:
            classifier, priority_scorer, workflow = init_system()
            st.success("✅ System ready!")
        except Exception as e:
            st.error(f"❌ Failed to initialize: {e}")
            st.info("Make sure HUGGINGFACE_API_KEY is set in your .env file")
            return
    
    # Main content
    tab_input, tab_samples = st.tabs(["📝 Enter Incident", "📋 Sample Incidents"])
    
    with tab_input:
        st.markdown("### Enter Incident Report")
        
        report_text = st.text_area(
            "Describe the incident in detail:",
            height=150,
            placeholder="Example: Patient in Room 312 received wrong medication. Nurse administered Metoprolol 100mg instead of prescribed Metformin 500mg..."
        )
        
        # Patient Demographics Section
        st.markdown("### 👤 Patient Demographics (Optional)")
        st.caption("Providing demographics helps identify vulnerable populations and improve prioritization.")
        
        demo_col1, demo_col2, demo_col3 = st.columns(3)
        
        with demo_col1:
            patient_age = st.number_input("Age", min_value=0, max_value=120, value=None, placeholder="Enter age")
            patient_gender = st.selectbox(
                "Gender",
                options=["", "Male", "Female", "Non-binary", "Transgender Male", "Transgender Female", "Other", "Prefer not to say"],
                index=0
            )
        
        with demo_col2:
            patient_race = st.selectbox(
                "Race",
                options=["", "White", "Black/African American", "Asian", "Hispanic/Latino", 
                        "Native American/Alaska Native", "Native Hawaiian/Pacific Islander", 
                        "Multiracial", "Other", "Unknown"],
                index=0
            )
            patient_ethnicity = st.selectbox(
                "Ethnicity",
                options=["", "Hispanic", "Non-Hispanic", "Unknown"],
                index=0
            )
        
        with demo_col3:
            patient_language = st.selectbox(
                "Primary Language",
                options=["English", "Spanish", "Mandarin", "Vietnamese", "Korean", 
                        "Tagalog", "Arabic", "Russian", "Other"],
                index=0
            )
            patient_insurance = st.selectbox(
                "Insurance",
                options=["", "Medicare", "Medicaid", "Private Insurance", "Self-Pay", 
                        "Uninsured", "Tricare", "VA"],
                index=0
            )
        
        # Build demographics dict if any field is filled
        demographics = None
        if any([patient_age, patient_gender, patient_race, patient_ethnicity, patient_language != "English", patient_insurance]):
            demographics = {
                "age": patient_age if patient_age else "Unknown",
                "gender": patient_gender if patient_gender else "Unknown",
                "race": patient_race if patient_race else "Unknown",
                "ethnicity": patient_ethnicity if patient_ethnicity else "Unknown",
                "language": patient_language,
                "insurance": patient_insurance if patient_insurance else "Unknown"
            }
        
        st.markdown("---")
        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_btn = st.button("🔍 Analyze Incident", type="primary", use_container_width=True)
        
        if analyze_btn and report_text:
            with st.spinner("🤖 Analyzing with Llama 3.3..."):
                result = process_incident(classifier, priority_scorer, workflow, report_text, demographics=demographics)
            
            st.markdown("---")
            display_results(result)
        
        elif analyze_btn:
            st.warning("Please enter an incident report to analyze.")
    
    with tab_samples:
        st.markdown("### Sample Incidents")
        st.write("Click on a sample to analyze it:")
        
        samples = get_all_sample_incidents()[:6]  # Show first 6 samples
        
        for i, sample in enumerate(samples):
            with st.expander(f"📄 {sample['id']}: {sample['text'][:80]}..."):
                st.write(sample["text"])
                
                # Show demographics if available
                demo = sample.get('patient_demographics')
                if demo:
                    st.caption(f"Patient: {demo.get('age', '?')}yo {demo.get('gender', '?')}, {demo.get('race', '?')}, {demo.get('language', 'English')}-speaking, {demo.get('insurance', '?')}")
                
                if st.button(f"Analyze {sample['id']}", key=f"sample_{i}"):
                    with st.spinner("🤖 Analyzing with Llama 3.3..."):
                        result = process_incident(
                            classifier, priority_scorer, workflow, 
                            sample["text"], sample["id"],
                            sample.get('patient_demographics')
                        )
                    display_results(result)


if __name__ == "__main__":
    main()
