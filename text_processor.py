from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
import re
import spacy

class QuestionType(Enum):
    GREETING = "greeting"
    LOCATION = "location"
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    COMPARATIVE = "comparative"
    UNKNOWN = "unknown"

class TextProcessor:
    def __init__(self):
        """Initialize the text processor with NLP model and patterns."""
        self.nlp = spacy.load('en_core_web_sm')
        
        # Common variations of question patterns
        self.question_patterns = {
            QuestionType.LOCATION: [
                r'where (?:is|are|can i find) (?:the )?(.+)',
                r'(?:location|address|directions?) (?:of|to|for) (?:the )?(.+)',
                r'how (?:do|can) (?:i|we) (?:get|reach|find) (?:the )?(.+)',
                r'show me (?:where|how to get to) (?:the )?(.+)'
            ],
            QuestionType.FACTUAL: [
                r'what (?:is|are) (?:the )?(.+)',
                r'tell me about (?:the )?(.+)',
                r'(?:information|details) (?:about|on|for) (?:the )?(.+)',
                r'describe (?:the )?(.+)'
            ],
            QuestionType.PROCEDURAL: [
                r'how (?:do|can|should) (?:i|we) (.+)',
                r'what (?:are|is) the (?:steps|process|procedure) (?:to|for) (.+)',
                r'guide (?:me|us) (?:through|on) (.+)',
                r'explain how to (.+)'
            ],
            QuestionType.COMPARATIVE: [
                r'(?:compare|difference between) (.+)',
                r'which is (?:better|worse|more|less) (.+)',
                r'what are the (?:pros|cons|advantages|disadvantages) of (.+)',
                r'how does (.+) compare to (.+)'
            ]
        }
        
        # Common entity patterns
        self.entity_patterns = {
            'campus': [
                r'(?:srm|srmist)?\s*(?:university|institute|college)?\s*(?:campus|branch)?\s*(?:at|in)?\s*(kattankulathur|chennai|delhi-ncr|modinagar|ramapuram|vadapalani|amaravati|sikkim)',
                r'(kattankulathur|delhi-ncr|modinagar|ramapuram|vadapalani|amaravati|sikkim)\s*(?:campus|branch)',
            ],
            'facility': [
                r'(?:the)?\s*(tech\s*park|library|hostel|cafeteria|sports\s*complex|gymnasium|labs?|auditorium)',
                r'(central\s*library|main\s*building|university\s*building|admin\s*block)',
            ],
            'program': [
                r'(?:b\.?tech|m\.?tech|bba|mba|phd|diploma)\s*(?:in)?\s*(.*?)\s*(?:program|course|degree)?',
                r'(?:bachelor|master)s?\s*(?:of|in)\s*(.*?)\s*(?:program|course|degree)?',
            ]
        }
        
        # Common greeting patterns
        self.greeting_patterns = [
            r'^(?:hi|hello|hey|greetings|good\s*(?:morning|afternoon|evening))(?:\s|$)',
            r'^(?:how\s*are\s*you|what\'s\s*up)(?:\s|$)'
        ]

    def classify_question(self, query: str) -> Dict[str, Any]:
        """
        Classify the question type and extract relevant entities using both
        pattern matching and semantic similarity.
        """
        query = query.lower().strip()
        
        # Check for greetings first
        if any(re.match(pattern, query) for pattern in self.greeting_patterns):
            return {
                'type': QuestionType.GREETING,
                'entities': [],
                'context': {}
            }
        
        # Process with spaCy for semantic understanding
        doc = self.nlp(query)
        
        # Try pattern matching first
        question_type = self._match_question_type(query)
        entities = self._extract_entities(query)
        
        # If pattern matching fails, use semantic similarity
        if question_type == QuestionType.UNKNOWN:
            question_type = self._semantic_question_classification(doc)
        
        # Extract additional context
        context = self._extract_context(doc)
        
        # If still no entities found, try semantic entity extraction
        if not entities:
            entities.extend(self._semantic_entity_extraction(doc))
        
        return {
            'type': question_type,
            'entities': entities,
            'context': context
        }

    def _match_question_type(self, query: str) -> QuestionType:
        """Match question type using regex patterns."""
        for qtype, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return qtype
        return QuestionType.UNKNOWN

    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities using regex patterns."""
        entities = []
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    if match.groups():
                        entities.append(match.group(1).strip())
        return list(set(entities))

    def _semantic_question_classification(self, doc) -> QuestionType:
        """Classify question type using semantic similarity."""
        # Example questions for each type
        type_examples = {
            QuestionType.LOCATION: ["Where is the library", "How do I get to the campus"],
            QuestionType.FACTUAL: ["What are the courses offered", "Tell me about the programs"],
            QuestionType.PROCEDURAL: ["How do I apply for admission", "What are the steps to register"],
            QuestionType.COMPARATIVE: ["Compare the campuses", "Which program is better"]
        }
        
        best_similarity = 0
        best_type = QuestionType.UNKNOWN
        
        for qtype, examples in type_examples.items():
            for example in examples:
                example_doc = self.nlp(example)
                similarity = doc.similarity(example_doc)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_type = qtype
        
        return best_type

    def _semantic_entity_extraction(self, doc) -> List[str]:
        """Extract entities using spaCy's NER and noun chunks."""
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'FAC', 'LOC', 'GPE']:
                entities.append(ent.text)
        
        # Extract noun chunks as potential entities
        for chunk in doc.noun_chunks:
            # Filter out common words and pronouns
            if not any(token.is_stop or token.pos_ == 'PRON' for token in chunk):
                entities.append(chunk.text)
        
        return list(set(entities))

    def _extract_context(self, doc) -> Dict[str, Any]:
        """Extract additional context from the query."""
        context = {
            'time_reference': None,
            'location_reference': None,
            'comparison_aspects': [],
            'requirements': []
        }
        
        for token in doc:
            # Extract time references
            if token.pos_ == 'NOUN' and token.text in ['today', 'tomorrow', 'yesterday']:
                context['time_reference'] = token.text
            
            # Extract location references
            if token.ent_type_ in ['GPE', 'LOC']:
                context['location_reference'] = token.text
            
            # Extract comparison aspects
            if token.dep_ == 'attr' and token.head.pos_ == 'ADJ':
                context['comparison_aspects'].append(token.text)
            
            # Extract requirements
            if token.text in ['need', 'require', 'must', 'should']:
                for child in token.children:
                    if child.dep_ == 'dobj':
                        context['requirements'].append(child.text)
        
        return context

    def format_response(self, question_type: QuestionType, data: Dict[str, Any]) -> str:
        """Format the response based on the question type and data."""
        if question_type == QuestionType.GREETING:
            return (
                "Hello! I'm ORB AI, your SRMIST Knowledge Assistant. I can help you with:\n"
                "- Campus locations and facilities\n"
                "- Academic programs and courses\n"
                "- Admission procedures\n"
                "- Student facilities\n\n"
                "What would you like to know about?"
            )
        
        if question_type == QuestionType.LOCATION:
            if not data:
                return "I'm sorry, I couldn't find information about that location."
            
            # Remove duplicates by using a dictionary with entity IDs as keys
            unique_items = {}
            for item in data:
                if 'id' in item:
                    unique_items[item['id']] = item
            
            response = []
            for item in unique_items.values():
                if 'address' in item:
                    response.append(f"Address: {item['address']}")
                if 'description' in item:
                    response.append(f"Description: {item['description']}")
                if 'facilities' in item:
                    if isinstance(item['facilities'], list):
                        # Handle both string and dictionary facilities
                        facility_names = []
                        for facility in item['facilities']:
                            if isinstance(facility, dict):
                                facility_names.append(facility.get('name', facility.get('id', '')))
                            else:
                                facility_names.append(str(facility))
                        response.append(f"Facilities: {', '.join(facility_names)}")
                    else:
                        response.append(f"Facilities: {item['facilities']}")
                if 'map_link' in item:
                    response.append(f"Map: {item['map_link']}")
            
            return "\n\n".join(response)
        
        if question_type == QuestionType.FACTUAL:
            if not data:
                return "I'm sorry, I don't have that information. Could you please rephrase your question?"
            
            response = []
            for key, value in data.items():
                if isinstance(value, list):
                    response.append(f"{key.title()}: {', '.join(str(v) for v in value)}")
                elif isinstance(value, dict):
                    details = [f"{k}: {v}" for k, v in value.items() if k != 'id']
                    response.append(f"{key.title()}:\n- " + "\n- ".join(details))
                else:
                    response.append(f"{key.title()}: {value}")
            
            return "\n\n".join(response)
        
        if question_type == QuestionType.PROCEDURAL:
            if not data.get('steps'):
                return "I'm sorry, I don't have step-by-step information for that. Could you please be more specific?"
            
            response = ["Here's what you need to do:"]
            for i, step in enumerate(data['steps'], 1):
                response.append(f"{i}. {step}")
            
            if 'additional_info' in data:
                response.append("\nAdditional Information:")
                response.append(data['additional_info'])
            
            return "\n".join(response)
        
        if question_type == QuestionType.COMPARATIVE:
            if not data.get('comparisons'):
                return "I'm sorry, I don't have enough information to make a comparison. Could you please specify what aspects you'd like to compare?"
            
            response = ["Here's a comparison:"]
            for item, details in data['comparisons'].items():
                response.append(f"\n{item}:")
                if isinstance(details, dict):
                    for key, value in details.items():
                        if key != 'id':
                            response.append(f"- {key}: {value}")
                else:
                    response.append(f"- {details}")
            
            return "\n".join(response)
        
        return "I'm sorry, I don't understand that type of question yet. Could you please rephrase it?" 