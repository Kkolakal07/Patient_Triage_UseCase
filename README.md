# Patient Complaint & Incident Report Triage System

An AI-powered system for automatically classifying, prioritizing, and routing patient complaints and incident reports in healthcare settings.

## Features

- **🏷️ Automated Classification**: Categorizes incidents by type (medication error, falls, infections, etc.), department, and severity level
- **⚡ Priority Scoring**: Computes urgency scores based on severity, recurrence likelihood, patient impact, and regulatory risk
- **📬 Smart Routing**: Routes incidents to appropriate personnel with escalation paths
- **🔔 Notifications**: Multi-channel alerting (email, SMS, pager, dashboard)
- **📊 Human-in-the-Loop**: Flags low-confidence classifications for review

## Quick Start

### Basic Usage (No Dependencies Required)

```bash
cd patient_triage_system

# Run demo with sample incidents
python main.py

# Interactive mode - enter your own incident reports
python main.py --interactive
```

### Sample Output

```
INCIDENT: INC-2024-003
═══════════════════════════════════════════════════════════════

📋 REPORT SUMMARY:
   Surgical team performed procedure on wrong knee...

🏷️ CLASSIFICATION:
   Category:    SURGICAL_ERROR
   Department:  surgery
   Severity:    SENTINEL_EVENT
   Confidence:  95%

⚡ PRIORITY:
   Score:       9.8/10
   Urgency:     CRITICAL
   SLA:         1 hours

📬 ROUTING:
   Primary:     chief_medical_officer
   CC:          risk_management, legal, compliance
   Notify via:  pager, sms, email, dashboard
   Immediate:   🚨 YES
```

## Project Structure

```
patient_triage_system/
├── main.py              # Main entry point & demo
├── classifier.py        # NLP classification module
├── priority_scorer.py   # Priority scoring engine
├── router.py            # Routing & notification engine
├── models.py            # Data models & enums
├── config.py            # Configuration settings
├── sample_data.py       # Sample incidents for testing
├── api.py               # REST API (optional)
└── requirements.txt     # Dependencies
```

## Components

### 1. Classifier (`classifier.py`)

- Pattern-based classification using keyword matching
- Optional transformer-based classification using ClinicalBERT
- Entity extraction (medications, staff roles, times, locations)
- Sentiment analysis for patient distress

### 2. Priority Scorer (`priority_scorer.py`)

Computes weighted priority score:

| Component | Weight | Description |
|-----------|--------|-------------|
| Severity | 40% | Based on harm level (near-miss to sentinel) |
| Recurrence | 20% | Pattern detection from historical data |
| Patient Impact | 20% | Vulnerability and distress level |
| Regulatory | 20% | Compliance and reporting requirements |

### 3. Router (`router.py`)

- Category-based routing rules
- Escalation path generation
- SLA monitoring
- Multi-channel notifications

## Configuration

Edit `config.py` to customize:

- **PRIORITY_WEIGHTS**: Adjust scoring component weights
- **ROUTING_RULES**: Define routing destinations by category
- **SEVERITY_KEYWORDS**: Keywords that trigger severity levels
- **SLA_BY_URGENCY**: Response time requirements

## API Server (Optional)

```bash
# Install dependencies
pip install fastapi uvicorn

# Start server
uvicorn api:app --reload

# Access docs at http://localhost:8000/docs
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/triage` | Process single incident |
| POST | `/api/v1/triage/batch` | Process multiple incidents |
| GET | `/api/v1/incidents/{id}` | Retrieve processed incident |
| GET | `/api/v1/stats` | System statistics |

## Advanced Features

### Transformer-Based Classification

For improved accuracy with clinical text:

```bash
pip install transformers torch

python main.py --advanced
```

### Integration Points

- **EHR Integration**: Submit incidents via REST API
- **Email Ingestion**: Parse incoming complaint emails
- **Dashboard**: Real-time incident monitoring
- **Analytics**: Export data for trend analysis

## Incident Categories

| Category | Examples |
|----------|----------|
| `medication_error` | Wrong dose, wrong drug, adverse reaction |
| `patient_fall` | Falls, slips, bed injuries |
| `infection` | HAI, surgical site infection, sepsis |
| `surgical_error` | Wrong site, retained objects |
| `equipment_failure` | Device malfunction, alarm failure |
| `communication` | Handoff errors, language barriers |
| `privacy_breach` | HIPAA violations, unauthorized access |
| `patient_experience` | Complaints about service, wait times |

## Severity Levels

| Level | Description | SLA |
|-------|-------------|-----|
| `SENTINEL_EVENT` | Death or serious permanent harm | 1 hour |
| `SERIOUS_HARM` | Permanent harm or prolonged hospitalization | 4 hours |
| `MODERATE_HARM` | Temporary harm requiring intervention | 24 hours |
| `MINOR_HARM` | Temporary harm, no intervention | 72 hours |
| `NEAR_MISS` | No harm reached patient | 72 hours |

## Compliance Considerations

- **HIPAA**: De-identify PHI before processing
- **Audit Trail**: All decisions are logged
- **Explainability**: Classification reasoning available
- **Human Review**: Low-confidence cases flagged

## License

MIT License - See LICENSE file for details.
