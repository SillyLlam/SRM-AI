import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from nlu_module import NLUModule
from context_manager import ContextManager
from response_formatter import ResponseFormatter
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedChatbot:
    def __init__(self, knowledge_base_path: str = 'knowledge_base.json'):
        """Initialize the enhanced chatbot with all components."""
        # Load knowledge base
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        
        # Initialize components
        self.nlu = NLUModule()
        self.context_manager = ContextManager()
        self.response_formatter = ResponseFormatter()
        
        # Initialize conversation tracking
        self.active_sessions = {}

    def _load_knowledge_base(self, path: str) -> Dict:
        """Load the knowledge base from JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
            return {}

    def start_conversation(self) -> str:
        """Start a new conversation and return session ID."""
        session_id = str(uuid.uuid4())
        self.context_manager.create_conversation(session_id)
        return session_id

    def process_query(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Process a user query and return a response."""
        try:
            # Create session if not provided
            if not session_id:
                session_id = self.start_conversation()
            
            # Get current context
            context = self.context_manager.get_context(session_id)
            
            # Analyze the query
            analysis = self.nlu.analyze_query(query, context)
            logger.info(f"Query analysis: {analysis}")
            
            # Check if it's a follow-up question
            if analysis['is_followup'] and context:
                response = self._handle_followup(query, analysis, context, session_id)
            else:
                # Find best matching information
                response = self._find_best_match(analysis, session_id)
            
            # Update conversation context
            self.context_manager.update_context(
                session_id=session_id,
                query=query,
                response=response,
                entities=analysis.get('entities', []),
                intent=analysis.get('intent')
            )
            
            # Format the response
            formatted_response = self.response_formatter.format_response(response)
            
            # Add conversation metadata
            formatted_response['session_id'] = session_id
            formatted_response['conversation_summary'] = self.context_manager.get_conversation_summary(session_id)
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                'type': 'error',
                'message': "I'm sorry, but I encountered an error processing your request.",
                'confidence': 0.0
            }

    def _handle_followup(self, query: str, analysis: Dict, context: Dict, session_id: str) -> Dict[str, Any]:
        """Handle follow-up questions using conversation context."""
        # Get the current entity and intent from context
        current_entity = context.get('current_entity')
        current_intent = context.get('current_intent')
        
        # If we have a current entity, try to answer about it
        if current_entity:
            # Determine the new intent for the follow-up
            new_intent = analysis['intent'] if analysis['intent'] != 'general' else current_intent
            
            # Find information about the current entity with the new intent
            response = self._get_entity_info(current_entity, new_intent)
            if response.get('confidence', 0) > 0.6:
                return response
        
        # If follow-up handling failed, try regular matching
        return self._find_best_match(analysis, session_id)

    def _find_best_match(self, analysis: Dict, session_id: str) -> Dict[str, Any]:
        """Find the best matching response based on query analysis."""
        entities = analysis.get('entities', [])
        intent = analysis.get('intent', 'general')
        
        # If we have entities, try to find exact matches
        if entities:
            for entity in entities:
                response = self._get_entity_info(entity, intent)
                if response.get('confidence', 0) > 0.6:
                    return response
        
        # If no exact matches, try semantic search
        query = analysis['processed_query']
        candidates = self._get_candidate_responses(query)
        
        if candidates:
            best_matches = self.nlu.find_best_matches(query, candidates)
            if best_matches:
                best_match, confidence = best_matches[0]
                if confidence > 0.6:
                    return self._format_candidate_response(best_match, confidence)
        
        # If no good matches, generate a fallback response
        return self._generate_fallback_response(analysis, session_id)

    def _get_entity_info(self, entity: str, intent: str) -> Dict[str, Any]:
        """Get information about an entity based on intent."""
        # Search for the entity in the knowledge base
        for category, items in self.knowledge_base.items():
            if entity in items:
                entity_data = items[entity]
                
                # Prepare response based on intent
                response = {
                    'type': intent,
                    'entity': entity,
                    'confidence': 1.0
                }
                
                # Add relevant information based on intent
                if intent == 'location':
                    response.update({
                        'address': entity_data.get('address'),
                        'location': entity_data.get('location'),
                        'map_link': entity_data.get('map_link')
                    })
                elif intent == 'description':
                    response.update({
                        'description': entity_data.get('description'),
                        'type_info': entity_data.get('type'),
                        'category': category
                    })
                elif intent == 'facilities':
                    response.update({
                        'facilities': entity_data.get('facilities', []) + entity_data.get('amenities', []),
                        'description': entity_data.get('description')
                    })
                elif intent == 'contact':
                    response.update({
                        'contact': entity_data.get('contact'),
                        'additional_info': entity_data.get('description')
                    })
                else:
                    # For other intents, include all relevant information
                    response.update({k: v for k, v in entity_data.items() 
                                  if k not in ['id', 'type'] and not isinstance(v, dict)})
                
                return response
        
        return {'confidence': 0.0}

    def _get_candidate_responses(self, query: str) -> List[str]:
        """Get candidate responses for semantic matching."""
        candidates = []
        
        # Generate candidate responses from knowledge base
        for category, items in self.knowledge_base.items():
            for entity, data in items.items():
                # Add entity name and description
                candidates.append(f"{entity}: {data.get('description', '')}")
                
                # Add other relevant information
                if 'facilities' in data:
                    candidates.append(f"{entity} facilities: {', '.join(data['facilities'])}")
                if 'address' in data:
                    candidates.append(f"{entity} is located at {data['address']}")
                if 'contact' in data:
                    contact_info = data['contact']
                    if isinstance(contact_info, dict):
                        candidates.append(f"Contact {entity}: " + 
                                       ', '.join(f"{k}: {v}" for k, v in contact_info.items()))
                    else:
                        candidates.append(f"Contact {entity}: {contact_info}")
        
        return candidates

    def _format_candidate_response(self, candidate: str, confidence: float) -> Dict[str, Any]:
        """Format a candidate response with metadata."""
        # Split entity and information
        parts = candidate.split(':', 1)
        entity = parts[0].strip()
        info = parts[1].strip() if len(parts) > 1 else ""
        
        # Determine response type based on content
        if 'located at' in candidate:
            response_type = 'location'
        elif 'facilities' in candidate:
            response_type = 'facilities'
        elif 'Contact' in candidate:
            response_type = 'contact'
        else:
            response_type = 'description'
        
        return {
            'type': response_type,
            'entity': entity,
            'description': info,
            'confidence': confidence
        }

    def _generate_fallback_response(self, analysis: Dict, session_id: str) -> Dict[str, Any]:
        """Generate a fallback response with suggestions."""
        # Get conversation context
        context = self.context_manager.get_context(session_id)
        
        # Try to find similar entities based on the query
        similar_entities = []
        if analysis.get('entities'):
            for entity in analysis['entities']:
                # Get embeddings for the entity
                entity_embedding = self.nlu.get_query_embedding(entity)
                
                # Compare with all entities in knowledge base
                for category, items in self.knowledge_base.items():
                    for kb_entity in items.keys():
                        similarity = self.nlu.get_semantic_similarity(entity, kb_entity)
                        if similarity > 0.5:  # Threshold for similarity
                            similar_entities.append((kb_entity, similarity))
        
        # Sort similar entities by similarity score
        similar_entities.sort(key=lambda x: x[1], reverse=True)
        
        # Generate suggestions based on similar entities
        suggestions = []
        if similar_entities:
            for entity, _ in similar_entities[:3]:
                suggestions.extend([
                    f"What is {entity}?",
                    f"Tell me about {entity}",
                    f"Where is {entity} located?"
                ])
        else:
            # If no similar entities found, suggest popular topics
            suggestions = [
                "Tell me about the Kattankulathur Campus",
                "What facilities are available in Tech Park?",
                "How can I contact the admissions office?",
                "What are the hostel facilities?",
                "Where is the Central Library?"
            ]
        
        return {
            'type': 'fallback',
            'message': "I'm not quite sure about that. Here are some related questions you might be interested in:",
            'suggestions': suggestions[:5],  # Limit to top 5 suggestions
            'confidence': 0.0
        }

    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of the conversation."""
        return self.context_manager.get_conversation_summary(session_id) 