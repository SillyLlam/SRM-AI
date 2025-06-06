from typing import Dict, Any, Optional, List, Tuple
from knowledge_graph import SRMKnowledgeGraph
from text_processor import TextProcessor, QuestionType
from admission_handler import AdmissionHandler
from datetime import datetime
from flask import Flask, request, jsonify
from semantic_search import SemanticSearchEngine
import logging
import traceback

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ORBAI:
    def __init__(self):
        self.knowledge_graph = SRMKnowledgeGraph()
        self.text_processor = TextProcessor()
        self.admission_handler = AdmissionHandler()
        
        # Enhanced conversation context
        self.context = {
            'last_question_type': None,
            'last_entities': None,
            'current_topic': None,
            'conversation_history': [],
            'referenced_entities': set(),
            'follow_up_context': {},
            'clarification_needed': False
        }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query and return a formatted response."""
        # Store the query in conversation history
        self.context['conversation_history'].append({
            'query': query,
            'timestamp': datetime.now().isoformat()
        })
        
        # Classify the question and extract entities
        classification = self.text_processor.classify_question(query)
        question_type = classification['type']
        entities = classification['entities']
        query_context = classification['context']
        
        # Update conversation context
        self.context['last_question_type'] = question_type
        self.context['last_entities'] = entities
        
        # Handle follow-up questions
        if self._is_follow_up_question(query, classification):
            entities = self._resolve_references(entities, query)
            
        # Process based on question type
        response = self._process_by_type(question_type, entities, query_context)
        
        # If response is empty, try fallback strategies
        if not response or (not response.get('information') and not response.get('formatted_answer')):
            response = self._handle_fallback(query, classification)
        
        # Update context with the response
        self.context['conversation_history'][-1]['response'] = response
        
        return response
    
    def _is_follow_up_question(self, query: str, classification: Dict[str, Any]) -> bool:
        """Detect if the current query is a follow-up question."""
        # Check for pronouns referring to previous entities
        pronouns = ['it', 'this', 'that', 'these', 'those', 'they', 'there']
        query_lower = query.lower()
        
        has_pronouns = any(pronoun in query_lower.split() for pronoun in pronouns)
        has_no_entities = not classification['entities']
        has_previous_context = bool(self.context['last_entities'])
        
        return has_pronouns or (has_no_entities and has_previous_context)
    
    def _resolve_references(self, current_entities: List[str], query: str) -> List[str]:
        """Resolve entity references using conversation context."""
        resolved_entities = list(current_entities)
        
        # If no current entities but we have previous context
        if not current_entities and self.context['last_entities']:
            # Add relevant previous entities based on query type
            if any(word in query.lower() for word in ['same', 'this', 'that', 'it']):
                resolved_entities.extend(self.context['last_entities'])
        
        # Add entities from follow-up context if relevant
        if self.context['follow_up_context'].get('relevant_entities'):
            resolved_entities.extend(
                self.context['follow_up_context']['relevant_entities']
            )
        
        return list(set(resolved_entities))
    
    def _process_by_type(
        self, 
        question_type: QuestionType, 
        entities: List[str], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process the query based on its type and context."""
        if question_type == QuestionType.GREETING:
            return {
                'type': 'greeting',
                'formatted_answer': self.text_processor.format_response(question_type, {})
            }
        
        if question_type == QuestionType.LOCATION:
            return self._handle_location_query(entities, context)
        
        if question_type == QuestionType.FACTUAL:
            return self._handle_factual_query(entities, context)
        
        if question_type == QuestionType.PROCEDURAL:
            return self._handle_procedural_query(entities, context)
        
        if question_type == QuestionType.COMPARATIVE:
            return self._handle_comparative_query(entities, context)
        
        return {
            'type': 'unknown',
            'formatted_answer': "I'm sorry, I don't understand that type of question yet."
        }
    
    def _handle_location_query(
        self, 
        entities: List[str], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle location-based queries with enhanced context."""
        response_data = []
        
        for entity in entities:
            # Check campuses first
            campus_info = self.knowledge_graph.query('campus', {'id': entity})
            if campus_info:
                response_data.extend(campus_info)
                # Store for potential follow-up questions
                self.context['follow_up_context']['campus'] = entity
                continue
            
            # Check specific locations
            location_info = self.knowledge_graph.query('location', {'id': entity})
            if location_info:
                response_data.extend(location_info)
                # Store for potential follow-up questions
                self.context['follow_up_context']['location'] = entity
                continue
            
            # Try text search if exact match not found
            search_results = self.knowledge_graph.search_by_text(entity)
            if search_results:
                response_data.extend(search_results)
                # Store search terms for context
                self.context['follow_up_context']['search_terms'] = entity
        
        return {
            'type': 'location',
            'information': response_data,
            'formatted_answer': self.text_processor.format_response(
                QuestionType.LOCATION, 
                response_data
            )
        }
    
    def _handle_factual_query(
        self, 
        entities: List[str], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle factual queries with enhanced context awareness."""
        response_data = {}
        
        # If no specific entities found, try to extract from query
        if not entities and context.get('location_reference'):
            entities = [context['location_reference']]
        
        if entities:
            # Process each identified entity
            for entity in entities:
                # Try exact matches first
                for entity_type in ['campus', 'location', 'program', 'facility']:
                    results = self.knowledge_graph.query(entity_type, {'id': entity})
                    if results:
                        response_data[entity_type] = results
                        # Store for follow-up questions
                        self.context['follow_up_context'][entity_type] = entity
                        break
                
                # If no exact matches, try text search
                if entity not in str(response_data):
                    search_results = self.knowledge_graph.search_by_text(entity)
                    if search_results:
                        for result in search_results:
                            entity_type = result.get('type', 'unknown')
                            if entity_type not in response_data:
                                response_data[entity_type] = []
                            response_data[entity_type].append(result)
        else:
            # Try to use context from previous queries
            if self.context['last_entities']:
                for entity in self.context['last_entities']:
                    results = self.knowledge_graph.search_by_text(entity)
                    if results:
                        response_data['related_info'] = results
        
        return {
            'type': 'factual',
            'information': response_data,
            'formatted_answer': self.text_processor.format_response(
                QuestionType.FACTUAL, 
                response_data
            )
        }
    
    def _handle_procedural_query(
        self, 
        entities: List[str], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle procedural queries with step-by-step information."""
        steps = []
        additional_info = None
        
        # Check for navigation/direction queries
        if any(word in ' '.join(entities).lower() for word in ['reach', 'get to', 'direction']):
            steps, additional_info = self._get_navigation_steps(entities)
        
        # Check for admission-related queries
        elif any(word in ' '.join(entities).lower() for word in ['admission', 'apply', 'join']):
            steps, additional_info = self._get_admission_steps(entities, context)
        
        # Handle other types of procedures
        else:
            steps, additional_info = self._get_general_procedure(entities, context)
        
        response = {
            'type': 'procedural',
            'steps': steps
        }
        
        if additional_info:
            response['additional_info'] = additional_info
        
        response['formatted_answer'] = self.text_processor.format_response(
            QuestionType.PROCEDURAL, 
            response
        )
        
        return response
    
    def _handle_comparative_query(
        self, 
        entities: List[str], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle comparative queries with detailed analysis."""
        comparisons = {}
        comparison_aspects = context.get('comparison_aspects', [])
        
        # If no specific aspects mentioned, use default comparison points
        if not comparison_aspects:
            comparison_aspects = ['location', 'facilities', 'programs']
        
        for entity in entities:
            entity_info = None
            
            # Try to find information about the entity
            for entity_type in ['campus', 'program', 'facility']:
                results = self.knowledge_graph.query(entity_type, {'id': entity})
                if results:
                    entity_info = results[0]
                    # Get related information based on comparison aspects
                    for aspect in comparison_aspects:
                        related = self.knowledge_graph.get_related_entities(
                            entity,
                            aspect
                        )
                        if related:
                            if 'related' not in entity_info:
                                entity_info['related'] = {}
                            entity_info['related'][aspect] = related
                    break
            
            if entity_info:
                comparisons[entity] = entity_info
        
        return {
            'type': 'comparative',
            'comparisons': comparisons,
            'formatted_answer': self.text_processor.format_response(
                QuestionType.COMPARATIVE,
                {'comparisons': comparisons}
            )
        }
    
    def _handle_fallback(
        self, 
        query: str, 
        classification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle cases where no direct answer is found."""
        # Try to find similar questions
        similar_questions = self.find_similar_questions(query)
        
        if similar_questions:
            return {
                'type': 'suggestion',
                'formatted_answer': (
                    "I'm not sure about that, but you might be interested in:\n" +
                    "\n".join(f"- {q}" for q in similar_questions)
                )
            }
        
        # If this is a follow-up question, try to use context
        if self._is_follow_up_question(query, classification):
            if self.context['last_entities']:
                return {
                    'type': 'clarification',
                    'formatted_answer': (
                        "I'm not sure what you're asking about. "
                        "Are you still asking about " +
                        ", ".join(self.context['last_entities']) + "?"
                    )
                }
        
        # General fallback response
        return {
            'type': 'fallback',
            'formatted_answer': (
                "I apologize, but I couldn't find specific information about that. "
                "Could you please:\n"
                "1. Rephrase your question\n"
                "2. Be more specific\n"
                "3. Ask about a different topic like campus locations, programs, or facilities?"
            )
        }
    
    def _get_navigation_steps(
        self, 
        entities: List[str]
    ) -> Tuple[List[str], Optional[str]]:
        """Get navigation steps for a location."""
        steps = []
        additional_info = None
        
        for entity in entities:
            location_info = None
            
            # Check campuses first
            campus_results = self.knowledge_graph.query('campus', {'id': entity})
            if campus_results:
                location_info = campus_results[0]
            else:
                # Check other locations
                location_results = self.knowledge_graph.query('location', {'id': entity})
                if location_results:
                    location_info = location_results[0]
            
            if location_info:
                steps = [
                    f"The {entity} is located at: {location_info.get('address', 'Address not available')}",
                    "You can reach here by:",
                    "- Public Transport: Available bus and train services",
                    "- College Bus: Regular shuttle service from major points",
                    "- Private Transport: Well-connected by road"
                ]
                
                if 'map_link' in location_info:
                    additional_info = f"For directions, visit: {location_info['map_link']}"
                break
        
        return steps, additional_info
    
    def _get_admission_steps(
        self, 
        entities: List[str],
        context: Dict[str, Any]
    ) -> Tuple[List[str], Optional[str]]:
        """Get admission procedure steps."""
        steps = [
            "Visit the official SRM website (www.srmist.edu.in)",
            "Click on 'Admissions' section",
            "Choose your preferred program",
            "Fill out the online application form",
            "Pay the application fee",
            "Submit required documents",
            "Wait for the entrance exam date",
            "Appear for counseling if selected"
        ]
        
        additional_info = None
        
        # Check for specific program or type of admission
        if any('international' in entity.lower() for entity in entities):
            additional_info = (
                "For international admissions, additional documents required:\n"
                "- Passport copy\n"
                "- Previous academic records\n"
                "- English proficiency test scores"
            )
        elif any('transfer' in entity.lower() for entity in entities):
            additional_info = (
                "For transfer admissions:\n"
                "- Submit current institution transcripts\n"
                "- Obtain No Objection Certificate\n"
                "- Complete credit transfer evaluation"
            )
        
        return steps, additional_info
    
    def _get_general_procedure(
        self, 
        entities: List[str],
        context: Dict[str, Any]
    ) -> Tuple[List[str], Optional[str]]:
        """Get steps for general procedures."""
        steps = []
        additional_info = None
        
        # Handle common procedures
        if any('library' in entity.lower() for entity in entities):
            steps = [
                "Visit the library with your student ID",
                "Register at the front desk",
                "Get your library card",
                "Follow borrowing guidelines",
                "Return books on time"
            ]
            additional_info = "Library timings: 8:00 AM to 8:00 PM"
        
        elif any('hostel' in entity.lower() for entity in entities):
            steps = [
                "Submit hostel application",
                "Pay hostel fees",
                "Complete room allocation process",
                "Collect room keys",
                "Complete check-in formalities"
            ]
            additional_info = "Contact hostel office for more details"
        
        return steps, additional_info
    
    def _is_admission_query(self, text: str) -> bool:
        """Check if the query is related to admissions."""
        admission_keywords = [
            "admission", "apply", "application", "enroll",
            "join", "entrance", "exam", "srmjeee"
        ]
        return any(keyword in text for keyword in admission_keywords)
    
    def _handle_comparative_query(self, text: str, entities: Dict[str, list]) -> Dict[str, Any]:
        """Handle comparative questions (e.g., comparing programs across campuses)."""
        response = {"type": "comparative", "comparisons": []}
        
        # If comparing campuses
        if entities['campuses']:
            for campus in entities['campuses']:
                campus_info = self.knowledge_graph.query(
                    "campus",
                    {"id": campus}
                )
                if campus_info:
                    programs = self.knowledge_graph.get_related_entities(
                        campus,
                        "offers"
                    )
                    response["comparisons"].append({
                        "campus": campus,
                        "details": campus_info[0],
                        "programs": programs
                    })
        
        # If comparing programs
        elif entities['programs']:
            for program in entities['programs']:
                program_info = self.knowledge_graph.query(
                    "program",
                    {"id": program}
                )
                if program_info:
                    courses = self.knowledge_graph.get_related_entities(
                        program,
                        "includes"
                    )
                    response["comparisons"].append({
                        "program": program,
                        "details": program_info[0],
                        "courses": courses
                    })
        
        return response
    
    def _handle_procedural_query(self, text: str, entities: Dict[str, list]) -> Dict[str, Any]:
        """Handle procedural questions (how-to guides, steps, etc.)."""
        response = {"type": "procedural", "steps": []}
        
        # Example procedure for campus change
        if "change campus" in text or "campus transfer" in text:
            response["steps"] = [
                "1. Submit application to current campus office",
                "2. Obtain No Objection Certificate (NOC)",
                "3. Apply to target campus",
                "4. Wait for approval from both campuses",
                "5. Complete transfer formalities"
            ]
            response["contact"] = "transfer.office@srmist.edu.in"
        
        # Add more procedural handlers as needed
        
        return response
    
    def _handle_factual_query(self, text: str, entities: Dict[str, list]) -> Dict[str, Any]:
        """Handle factual questions (direct information queries)."""
        response = {"type": "factual", "information": {}}
        
        # First try direct entity queries
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                info = self.knowledge_graph.query(
                    entity_type.rstrip('s'),  # Remove plural 's'
                    {"id": entity}
                )
                if info:
                    response["information"][entity] = info[0]
        
        # If no specific entities found, try text search
        if not response["information"]:
            results = self.knowledge_graph.search_by_text(text)
            if results:
                for result in results:
                    entity_id = result.pop('id')
                    response["information"][entity_id] = result
                    
                    # If it's a location, get its facilities
                    if result.get('type') == 'location':
                        facilities = self.knowledge_graph.get_related_entities(entity_id)
                        if facilities:
                            response["information"][entity_id]['facilities'] = facilities
                    
                    # If it's a campus, get its locations and facilities
                    elif result.get('type') == 'campus':
                        locations = self.knowledge_graph.get_related_entities(
                            entity_id, 'has_location'
                        )
                        if locations:
                            response["information"][entity_id]['locations'] = locations
                        
                        facilities = self.knowledge_graph.get_related_entities(
                            entity_id, 'has_facility'
                        )
                        if facilities:
                            response["information"][entity_id]['facilities'] = facilities
        
        # Format the response for better readability
        if response["information"]:
            formatted_response = self._format_response(response["information"])
            response["formatted_answer"] = formatted_response
        
        return response
    
    def _format_response(self, information: Dict[str, Any]) -> str:
        """Format the response information into a readable string."""
        formatted = []
        
        for entity, details in information.items():
            if details.get('type') == 'campus':
                formatted.append(f"{entity} Campus:")
                formatted.append(f"- Location: {details.get('location', 'N/A')}")
                formatted.append(f"- Address: {details.get('address', 'N/A')}")
                
                if 'locations' in details:
                    formatted.append("- Notable locations:")
                    for loc in details['locations']:
                        formatted.append(f"  * {loc['id']}: {loc.get('description', '')}")
                
                if 'facilities' in details:
                    formatted.append("- Facilities available:")
                    for facility in details['facilities']:
                        if isinstance(facility.get('types', []), list):
                            formatted.append(f"  * {facility['id']}: {', '.join(facility['types'])}")
                        else:
                            formatted.append(f"  * {facility['id']}")
            
            elif details.get('type') == 'location':
                formatted.append(f"{entity}:")
                formatted.append(f"- {details.get('description', '')}")
                formatted.append(f"- Located in: {details.get('location', 'N/A')}")
                
                if 'facilities' in details:
                    formatted.append("- Available facilities:")
                    for facility in details['facilities']:
                        formatted.append(f"  * {facility}")
            
            elif details.get('type') == 'program':
                formatted.append(f"{entity} Program:")
                if 'degrees' in details:
                    formatted.append(f"- Degrees offered: {', '.join(details['degrees'])}")
                if 'departments' in details:
                    formatted.append("- Departments:")
                    for dept in details['departments']:
                        formatted.append(f"  * {dept}")
        
        return "\n".join(formatted)
    
    def find_similar_questions(self, query: str) -> List[str]:
        """Find similar questions based on the query."""
        suggestions = []
        
        # Convert query to lowercase for matching
        query = query.lower()
        
        # Common question patterns
        if 'where' in query:
            if 'srm' in query:
                suggestions.extend([
                    "What are the different SRM campuses?",
                    "Where is SRM Kattankulathur campus located?",
                    "How to reach SRM main campus?"
                ])
            if any(loc in query for loc in ['tech park', 'library', 'hostel']):
                suggestions.extend([
                    "What facilities are available in Tech Park?",
                    "Where is the Central Library located?",
                    "What are the hostel facilities?"
                ])
        
        elif 'how' in query:
            if any(word in query for word in ['admission', 'apply', 'join']):
                suggestions.extend([
                    "How to apply for admission at SRM?",
                    "What are the admission requirements?",
                    "How to apply for international admission?"
                ])
            if 'reach' in query:
                suggestions.extend([
                    "How to reach SRM from Chennai airport?",
                    "What transportation facilities are available?",
                    "Is there a college bus service?"
                ])
        
        elif any(word in query for word in ['program', 'course', 'degree']):
            suggestions.extend([
                "What programs are offered at SRM?",
                "Which engineering branches are available?",
                "What are the postgraduate programs?"
            ])
        
        elif any(word in query for word in ['facility', 'amenity', 'infrastructure']):
            suggestions.extend([
                "What facilities are available at SRM?",
                "What sports facilities are available?",
                "Tell me about the hostel facilities"
            ])
        
        # Return unique suggestions, limited to top 3
        return list(dict.fromkeys(suggestions))[:3]

# Initialize the semantic search engine
search_engine = SemanticSearchEngine()

def format_response(response_data):
    """Format the response based on its type."""
    response_type = response_data.get('type')
    
    if response_type == 'location':
        return {
            'response': (
                f"{response_data['entity']} is located at {response_data['address']}. "
                f"You can find it in {response_data['location']}.\n"
                f"Here's a map link: {response_data['map_link']}"
            ),
            'confidence': response_data.get('confidence', 1.0)
        }
    
    elif response_type == 'description':
        return {
            'response': (
                f"{response_data['entity']}: {response_data['description']}"
            ),
            'confidence': response_data.get('confidence', 1.0)
        }
    
    elif response_type == 'facilities':
        facilities_list = ", ".join(response_data['facilities'])
        return {
            'response': (
                f"{response_data['entity']} has the following facilities: {facilities_list}.\n"
                f"{response_data['description']}"
            ),
            'confidence': response_data.get('confidence', 1.0)
        }
    
    elif response_type == 'fallback':
        suggestions = "\n".join([f"- {q}" for q in response_data.get('suggestions', [])])
        return {
            'response': (
                f"{response_data['message']}\n\n"
                f"You might want to try these questions instead:\n{suggestions}"
            ),
            'confidence': 0.0
        }
    
    elif response_type == 'error':
        return {
            'response': response_data['message'],
            'confidence': 0.0
        }
    
    # Default full info response
    info = response_data.get('info', {})
    return {
        'response': f"Here's what I know about {info.get('id', 'this')}:\n" + 
                   "\n".join([f"{k}: {v}" for k, v in info.items() if k != 'id']),
        'confidence': response_data.get('confidence', 1.0)
    }

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'error': 'No message provided',
                'response': 'Please provide a message to process.'
            }), 400

        user_message = data['message']
        logger.info(f"Received message: {user_message}")

        # Get response from semantic search engine
        search_result = search_engine.search(user_message)
        formatted_response = format_response(search_result)

        logger.info(f"Generated response: {formatted_response}")
        return jsonify(formatted_response)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'response': "I'm sorry, but I encountered an error processing your request. Please try again."
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'ORB AI is running'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 