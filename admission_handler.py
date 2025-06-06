from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum

class AdmissionType(Enum):
    DOMESTIC = "domestic"
    INTERNATIONAL = "international"
    NRI = "nri"
    TRANSFER = "transfer"

@dataclass
class AdmissionRequirements:
    documents: list[str]
    eligibility: str
    contact_email: str
    procedure: str
    deadlines: Dict[str, str]

class AdmissionHandler:
    def __init__(self):
        self.admission_requirements = {
            AdmissionType.DOMESTIC: AdmissionRequirements(
                documents=[
                    "10th Mark Sheet",
                    "12th Mark Sheet",
                    "SRMJEEE Score Card",
                    "Aadhar Card",
                    "Passport size photographs"
                ],
                eligibility="Minimum 60% in PCM for Engineering",
                contact_email="admissions@srmist.edu.in",
                procedure="Apply through SRMJEEE and counselling",
                deadlines={
                    "SRMJEEE Registration": "April 30",
                    "Counselling": "June-July"
                }
            ),
            AdmissionType.INTERNATIONAL: AdmissionRequirements(
                documents=[
                    "High School Transcripts",
                    "Standardized Test Scores (SAT/ACT)",
                    "English Proficiency (IELTS/TOEFL)",
                    "Passport",
                    "Statement of Purpose"
                ],
                eligibility="Completed 12 years of education with good academic record",
                contact_email="admissions.ir@srmist.edu.in",
                procedure="Apply through International Admissions Portal",
                deadlines={
                    "Fall Semester": "June 30",
                    "Spring Semester": "December 15"
                }
            ),
            AdmissionType.NRI: AdmissionRequirements(
                documents=[
                    "NRI Status Proof",
                    "Passport copies",
                    "Academic transcripts",
                    "Bank statements"
                ],
                eligibility="NRI/NRI Sponsored candidates",
                contact_email="nri.admissions@srmist.edu.in",
                procedure="Direct admission through NRI quota",
                deadlines={
                    "Application": "May 31",
                    "Admission": "June 30"
                }
            ),
            AdmissionType.TRANSFER: AdmissionRequirements(
                documents=[
                    "Current University Transcripts",
                    "No Objection Certificate",
                    "Migration Certificate",
                    "Syllabus of completed courses"
                ],
                eligibility="Completed at least one year at recognized university",
                contact_email="transfer.admissions@srmist.edu.in",
                procedure="Apply with complete transcripts for credit transfer",
                deadlines={
                    "Fall Transfer": "July 15",
                    "Spring Transfer": "December 31"
                }
            )
        }
    
    def handle_admission_query(self, query: str) -> Dict[str, str]:
        """Process admission related queries and return relevant information."""
        query = query.lower()
        
        # Determine admission type
        admission_type = self._determine_admission_type(query)
        if not admission_type:
            return {
                "error": "Unable to determine admission type. Please specify if you're asking about domestic, international, NRI, or transfer admissions."
            }
        
        requirements = self.admission_requirements[admission_type]
        
        # Extract specific information based on query keywords
        response = {}
        
        if any(word in query for word in ["document", "require", "submit"]):
            response["documents"] = requirements.documents
        
        if any(word in query for word in ["eligible", "eligibility", "qualify"]):
            response["eligibility"] = requirements.eligibility
        
        if any(word in query for word in ["procedure", "process", "how to", "steps"]):
            response["procedure"] = requirements.procedure
        
        if any(word in query for word in ["deadline", "date", "when"]):
            response["deadlines"] = requirements.deadlines
        
        if any(word in query for word in ["contact", "email", "reach"]):
            response["contact"] = requirements.contact_email
        
        # If no specific information was requested, return everything
        if not response:
            response = {
                "documents": requirements.documents,
                "eligibility": requirements.eligibility,
                "procedure": requirements.procedure,
                "deadlines": requirements.deadlines,
                "contact": requirements.contact_email
            }
        
        return response
    
    def _determine_admission_type(self, query: str) -> Optional[AdmissionType]:
        """Determine the type of admission being asked about."""
        if any(word in query for word in ["international", "foreign", "abroad", "overseas"]):
            return AdmissionType.INTERNATIONAL
        elif any(word in query for word in ["nri", "non resident", "non-resident"]):
            return AdmissionType.NRI
        elif any(word in query for word in ["transfer", "change university", "credit transfer"]):
            return AdmissionType.TRANSFER
        elif any(word in query for word in ["domestic", "indian", "local", "srmjeee"]):
            return AdmissionType.DOMESTIC
        
        # Default to domestic if no specific type is mentioned
        return AdmissionType.DOMESTIC 