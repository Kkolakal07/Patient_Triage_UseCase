"""
Sample incident reports for testing the triage system.
Includes patient demographic data for comprehensive analysis.
"""

from datetime import datetime, timedelta
import random

SAMPLE_INCIDENTS = [
    {
        "id": "INC-2024-001",
        "text": """
        Patient in Room 312 received wrong medication. Nurse administered 
        Metoprolol 100mg instead of prescribed Metformin 500mg. Patient is 
        diabetic and began showing signs of hypoglycemia. Physician was 
        notified immediately. Patient required glucose administration and 
        monitoring. No permanent harm but patient was frightened and upset.
        Incident occurred during night shift around 2:00 AM.
        """,
        "reporter_role": "Charge Nurse",
        "location": "Medical Unit 3",
        "patient_demographics": {
            "age": 67,
            "gender": "Female",
            "race": "African American",
            "ethnicity": "Non-Hispanic",
            "language": "English",
            "insurance": "Medicare"
        }
    },
    {
        "id": "INC-2024-002", 
        "text": """
        Elderly patient (82 years old) found on floor next to bed. Call light 
        was not within reach. Patient has dementia and was attempting to get 
        to bathroom unassisted. Patient sustained hip fracture requiring 
        surgical intervention. Bed alarm was not activated despite fall risk 
        assessment indicating high risk. Family is extremely upset and 
        mentioned contacting a lawyer.
        """,
        "reporter_role": "RN",
        "location": "ICU",
        "patient_demographics": {
            "age": 82,
            "gender": "Male",
            "race": "White",
            "ethnicity": "Non-Hispanic",
            "language": "English",
            "insurance": "Medicare"
        }
    },
    {
        "id": "INC-2024-003",
        "text": """
        Surgical team performed procedure on wrong knee. Patient was scheduled 
        for left knee arthroscopy but right knee was prepped and operated on. 
        Time-out procedure was performed but surgeon did not verify surgical 
        site marking. Patient will require additional surgery on correct knee. 
        Patient and family devastated. This is a sentinel event requiring 
        immediate reporting to administration and state health department.
        """,
        "reporter_role": "OR Nurse",
        "location": "Operating Room 2",
        "patient_demographics": {
            "age": 45,
            "gender": "Male",
            "race": "Hispanic/Latino",
            "ethnicity": "Hispanic",
            "language": "Spanish",
            "insurance": "Private Insurance"
        }
    },
    {
        "id": "INC-2024-004",
        "text": """
        Patient complaint about wait time in Emergency Department. Patient 
        arrived with chest pain at 10:00 AM and was not seen by physician 
        until 2:30 PM. Patient reports being ignored by staff and sitting 
        in waiting room for hours. EKG eventually showed normal sinus rhythm 
        and patient was discharged with diagnosis of anxiety. Patient very 
        dissatisfied with care and communication.
        """,
        "reporter_role": "Patient Relations",
        "location": "Emergency Department",
        "patient_demographics": {
            "age": 54,
            "gender": "Female",
            "race": "Black/African American",
            "ethnicity": "Non-Hispanic",
            "language": "English",
            "insurance": "Medicaid"
        }
    },
    {
        "id": "INC-2024-005",
        "text": """
        IV pump malfunctioned during chemotherapy administration. Pump 
        delivered medication at incorrect rate for approximately 30 minutes 
        before alarm sounded. Patient received higher dose than prescribed. 
        Oncologist notified and patient placed on monitoring. Pump was 
        removed from service for biomedical inspection. Patient is 
        immunocompromised and family is concerned about long-term effects.
        """,
        "reporter_role": "Oncology RN",
        "location": "Oncology Unit",
        "patient_demographics": {
            "age": 58,
            "gender": "Female",
            "race": "Asian",
            "ethnicity": "Non-Hispanic",
            "language": "English",
            "insurance": "Private Insurance"
        }
    },
    {
        "id": "INC-2024-006",
        "text": """
        Two patients in post-surgical recovery developed surgical site 
        infections within 48 hours of procedure. Both patients had 
        appendectomies performed by same surgeon in same OR. Infection 
        control notified. Cultures pending. Suspicion of possible 
        contamination in OR or sterilization issue. Additional patients 
        who had surgery in that OR are being monitored.
        """,
        "reporter_role": "Infection Control",
        "location": "Surgery Department",
        "patient_demographics": {
            "age": 34,
            "gender": "Male",
            "race": "White",
            "ethnicity": "Non-Hispanic",
            "language": "English",
            "insurance": "Private Insurance"
        }
    },
    {
        "id": "INC-2024-007",
        "text": """
        Patient discovered their medical records were accessed by employee 
        who had no treatment relationship with them. Employee is a neighbor 
        of the patient. Patient extremely upset about privacy violation and 
        demanding action. IT audit confirms unauthorized access on three 
        separate occasions. This is a HIPAA violation requiring reporting.
        """,
        "reporter_role": "Privacy Officer",
        "location": "Administration",
        "patient_demographics": {
            "age": 41,
            "gender": "Female",
            "race": "White",
            "ethnicity": "Non-Hispanic",
            "language": "English",
            "insurance": "Private Insurance"
        }
    },
    {
        "id": "INC-2024-008",
        "text": """
        Near miss event: Pharmacist caught potential fatal drug interaction 
        before dispensing. Physician had ordered Warfarin for patient already 
        on Aspirin and Plavix without noting the interaction risk. Pharmacist 
        contacted physician who modified the order. No harm to patient but 
        highlights need for better medication reconciliation process.
        """,
        "reporter_role": "Pharmacist",
        "location": "Pharmacy",
        "patient_demographics": {
            "age": 72,
            "gender": "Male",
            "race": "White",
            "ethnicity": "Non-Hispanic",
            "language": "English",
            "insurance": "Medicare"
        }
    },
    {
        "id": "INC-2024-009",
        "text": """
        Patient with limited English proficiency did not receive interpreter 
        services during consent process for cardiac catheterization. Patient's 
        adult child served as interpreter but later family stated important 
        risks were not properly communicated. Procedure resulted in 
        complication (arterial damage) and family claims they would not have 
        consented if risks were properly explained. Cultural sensitivity and 
        language access concerns.
        """,
        "reporter_role": "Social Worker",
        "location": "Cardiology",
        "patient_demographics": {
            "age": 68,
            "gender": "Male",
            "race": "Asian",
            "ethnicity": "Non-Hispanic",
            "language": "Vietnamese",
            "insurance": "Medicare"
        }
    },
    {
        "id": "INC-2024-010",
        "text": """
        Code Blue called for infant in NICU. Respiratory monitor was found 
        disconnected. Unknown how long infant was without monitoring. Infant 
        resuscitated successfully but required increased respiratory support. 
        Parents are devastated. Investigation needed to determine if monitor 
        was accidentally disconnected or equipment malfunction. This is being 
        treated as a potential sentinel event pending investigation.
        """,
        "reporter_role": "NICU Charge Nurse",
        "location": "NICU",
        "patient_demographics": {
            "age": 0,  # 3 days old
            "gender": "Female",
            "race": "Black/African American",
            "ethnicity": "Non-Hispanic",
            "language": "N/A",
            "insurance": "Medicaid"
        }
    },
    {
        "id": "INC-2024-011",
        "text": """
        Patient complained that nurse was rude and dismissive when they 
        asked questions about their medications. Patient felt rushed and 
        said nurse rolled her eyes when patient asked about side effects.
        Patient otherwise satisfied with medical care but very upset about 
        the interaction. Requests that a different nurse be assigned.
        """,
        "reporter_role": "Patient",
        "location": "Medical Unit 2",
        "patient_demographics": {
            "age": 39,
            "gender": "Non-binary",
            "race": "Multiracial",
            "ethnicity": "Non-Hispanic",
            "language": "English",
            "insurance": "Private Insurance"
        }
    },
    {
        "id": "INC-2024-012",
        "text": """
        Lab specimen was lost requiring patient to return for repeat blood 
        draw. Patient has difficult IV access and procedure was painful.
        Lab cannot locate original specimen and no results were ever recorded.
        Patient frustrated as this delayed diagnosis of possible thyroid 
        condition by two weeks. Third incident this month involving lost 
        specimens from the same phlebotomist.
        """,
        "reporter_role": "Lab Supervisor",
        "location": "Laboratory",
        "patient_demographics": {
            "age": 29,
            "gender": "Female",
            "race": "Native American/Alaska Native",
            "ethnicity": "Non-Hispanic",
            "language": "English",
            "insurance": "Medicaid"
        }
    },
]


# Demographic options for generating synthetic data
DEMOGRAPHIC_OPTIONS = {
    "gender": ["Male", "Female", "Non-binary", "Transgender Male", "Transgender Female", "Other", "Prefer not to say"],
    "race": [
        "White", 
        "Black/African American", 
        "Asian", 
        "Hispanic/Latino",
        "Native American/Alaska Native", 
        "Native Hawaiian/Pacific Islander",
        "Multiracial",
        "Other",
        "Unknown"
    ],
    "ethnicity": ["Hispanic", "Non-Hispanic", "Unknown"],
    "language": ["English", "Spanish", "Mandarin", "Vietnamese", "Korean", "Tagalog", "Arabic", "Russian", "Other"],
    "insurance": ["Medicare", "Medicaid", "Private Insurance", "Self-Pay", "Uninsured", "Tricare", "VA"]
}


def generate_historical_incidents(count: int = 50) -> list:
    """
    Generate synthetic historical incidents for recurrence detection testing.
    """
    categories = [
        "medication_error", "patient_fall", "infection", 
        "equipment_failure", "communication", "patient_experience"
    ]
    departments = [
        "emergency", "surgery", "icu", "nursing", 
        "pharmacy", "radiology", "laboratory"
    ]
    locations = [
        "Unit 1", "Unit 2", "Unit 3", "OR 1", "OR 2", 
        "ED Bay 1", "ED Bay 2", "ICU Bed 1", "ICU Bed 2"
    ]
    
    incidents = []
    base_date = datetime.now() - timedelta(days=90)
    
    for i in range(count):
        incident_date = base_date + timedelta(days=random.randint(0, 90))
        incidents.append({
            "id": f"HIST-{i+1:04d}",
            "date": incident_date,
            "category": random.choice(categories),
            "department": random.choice(departments),
            "location": random.choice(locations),
            "severity": random.randint(1, 5),
        })
    
    # Add some clusters to test recurrence detection
    # Medication errors in pharmacy
    for i in range(5):
        incidents.append({
            "id": f"HIST-MED-{i+1:02d}",
            "date": datetime.now() - timedelta(days=random.randint(1, 30)),
            "category": "medication_error",
            "department": "pharmacy",
            "location": "Unit 2",
            "severity": 2,
        })
    
    # Falls in ICU
    for i in range(3):
        incidents.append({
            "id": f"HIST-FALL-{i+1:02d}",
            "date": datetime.now() - timedelta(days=random.randint(1, 20)),
            "category": "patient_fall",
            "department": "icu",
            "location": "ICU Bed 1",
            "severity": 3,
        })
    
    return incidents


def get_sample_incident(index: int = 0) -> dict:
    """Get a specific sample incident by index."""
    return SAMPLE_INCIDENTS[index % len(SAMPLE_INCIDENTS)]


def get_all_sample_incidents() -> list:
    """Get all sample incidents."""
    return SAMPLE_INCIDENTS.copy()
