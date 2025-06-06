from flask import Flask, request, jsonify
import re
from typing import Dict, Any, List, Tuple
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleChatbot:
    def __init__(self):
        # Knowledge base with structured information
        self.knowledge_base = {
            "Tech Park": {
                "id": "Tech Park",
                "type": "location",
                "description": "A state-of-the-art facility housing research labs and industry collaboration centers",
                "location": "Kattankulathur Campus",
                "address": "SRM Nagar, Kattankulathur, Chengalpattu District, Tamil Nadu - 603203",
                "facilities": ["Research Labs", "Innovation Center", "Industry Collaboration Space"],
                "map_link": "https://maps.app.goo.gl/HvLKqGK8TFE5QWLP6"
            },
            "Central Library": {
                "id": "Central Library",
                "type": "location",
                "description": "Multi-story library with vast collection of books, journals, and digital resources",
                "location": "Kattankulathur Campus",
                "facilities": ["Reading Halls", "Digital Library", "Conference Rooms"],
                "map_link": "https://maps.app.goo.gl/HvLKqGK8TFE5QWLP6"
            },
            "Kattankulathur Campus": {
                "id": "Kattankulathur",
                "type": "campus",
                "location": "Chennai",
                "established": "1985",
                "address": "SRM Nagar, Kattankulathur, Chengalpattu District, Tamil Nadu - 603203",
                "landmarks": ["Tech Park", "University Building", "Central Library"],
                "map_link": "https://maps.app.goo.gl/HvLKqGK8TFE5QWLP6"
            }
        }

        # Define patterns for different types of questions
        self.patterns = {
            'location': [
                r'where\s+is\s+(.*?)\??$',
                r'how\s+(?:do|can)\s+(?:I|we)\s+get\s+to\s+(.*?)\??$',
                r'(?:give|show)\s+(?:me)?\s+directions?\s+to\s+(.*?)\??$',
                r'what\s+is\s+the\s+location\s+of\s+(.*?)\??$'
            ],
            'description': [
                r'what\s+is\s+(.*?)\??$',
                r'tell\s+me\s+about\s+(.*?)\??$',
                r'describe\s+(.*?)\??$',
                r'give\s+(?:me)?\s+information\s+about\s+(.*?)\??$'
            ],
            'facilities': [
                r'what\s+facilities\s+(?:are|is)\s+(?:there\s+)?in\s+(.*?)\??$',
                r'what\s+can\s+(?:I|we)\s+find\s+in\s+(.*?)\??$',
                r'what(?:\'s)?\s+available\s+at\s+(.*?)\??$',
                r'what\s+does\s+(.*?)\s+have\??$'
            ]
        }

    def preprocess_query(self, query: str) -> str:
        """Basic query preprocessing."""
        # Convert to lowercase
        query = query.lower().strip()
        # Remove multiple spaces
        query = ' '.join(query.split())
        return query

    def find_best_match(self, query: str) -> Tuple[str, str, float]:
        """Find the best matching entity and intent for a query."""
        query = self.preprocess_query(query)
        
        # First try pattern matching
        for intent, pattern_list in self.patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, query)
                if match:
                    entity = match.group(1).strip()
                    # Find the closest matching entity in our knowledge base
                    best_entity = self.find_closest_entity(entity)
                    if best_entity:
                        return best_entity, intent, 1.0
        
        # If no pattern matches, try to find any mentioned entity
        for entity in self.knowledge_base:
            # Try exact match first
            if entity.lower() in query:
                # Try to determine intent from query keywords
                if any(word in query.lower() for word in ['where', 'location', 'address', 'map']):
                    return entity, 'location', 0.9
                elif any(word in query.lower() for word in ['facilities', 'available', 'have', 'contains']):
                    return entity, 'facilities', 0.9
                else:
                    return entity, 'description', 0.8
        
        # Try fuzzy matching as a last resort
        for entity in self.knowledge_base:
            entity_words = set(entity.lower().split())
            query_words = set(query.lower().split())
            if entity_words & query_words:  # If there's any word overlap
                return entity, 'description', 0.6
        
        return None, None, 0.0

    def find_closest_entity(self, query: str) -> str:
        """Find the closest matching entity in the knowledge base."""
        query = query.lower().strip()
        
        # First try exact match
        for entity in self.knowledge_base:
            if query == entity.lower():
                return entity
        
        # Then try case-insensitive contains
        for entity in self.knowledge_base:
            if query in entity.lower() or entity.lower() in query:
                return entity
            
        # Try matching individual words
        query_words = set(query.split())
        for entity in self.knowledge_base:
            entity_words = set(entity.lower().split())
            # If all words in the entity name are in the query
            if entity_words.issubset(query_words):
                return entity
            # If all words in the query are in the entity name
            if query_words.issubset(entity_words):
                return entity
        
        return None

    def get_response(self, entity: str, intent: str) -> Dict[str, Any]:
        """Generate a response based on the entity and intent."""
        data = self.knowledge_base.get(entity)
        if not data:
            return {
                'type': 'error',
                'message': f"I don't have information about {entity}."
            }

        if intent == 'location':
            return {
                'type': 'location',
                'entity': entity,
                'address': data.get('address'),
                'map_link': data.get('map_link'),
                'location': data.get('location')
            }
        elif intent == 'description':
            return {
                'type': 'description',
                'entity': entity,
                'description': data.get('description'),
                'type': data.get('type')
            }
        elif intent == 'facilities':
            return {
                'type': 'facilities',
                'entity': entity,
                'facilities': data.get('facilities', []),
                'description': data.get('description')
            }
        
        # Fallback to full information
        return {
            'type': 'full_info',
            'entity': entity,
            'info': data
        }

    def get_similar_questions(self, entity: str) -> List[str]:
        """Generate example questions for an entity."""
        examples = []
        if entity in self.knowledge_base:
            examples.extend([
                f"Where is {entity}?",
                f"What is {entity}?",
                f"Tell me about {entity}",
                f"What facilities are in {entity}?"
            ])
        return examples

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query and return a response."""
        entity, intent, confidence = self.find_best_match(query)
        
        if not entity or confidence < 0.6:
            # Try to find any mentioned entity for suggestions
            for known_entity in self.knowledge_base:
                if known_entity.lower() in query.lower():
                    suggestions = self.get_similar_questions(known_entity)
                    return {
                        'type': 'fallback',
                        'message': "I'm not quite sure what you're asking. Here are some questions you can try:",
                        'suggestions': suggestions
                    }
            
            # No entity found at all
            return {
                'type': 'fallback',
                'message': "I'm not sure what you're asking about. Here are some locations I know about:",
                'suggestions': [f"Tell me about {entity}" for entity in self.knowledge_base]
            }
        
        response = self.get_response(entity, intent)
        response['confidence'] = confidence
        return response

def format_response(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format the response based on its type."""
    try:
        response_type = response_data.get('type', 'unknown')
        
        if response_type == 'location':
            address = response_data.get('address', 'No address available')
            location = response_data.get('location', '')
            map_link = response_data.get('map_link', '')
            
            response_text = f"{response_data['entity']} is located at {address}."
            if location:
                response_text += f" You can find it in {location}."
            if map_link:
                response_text += f"\nHere's a map link: {map_link}"
            
            return {
                'response': response_text,
                'confidence': response_data.get('confidence', 1.0)
            }
        
        elif response_type == 'description':
            entity = response_data.get('entity', 'Unknown')
            description = response_data.get('description', 'No description available.')
            return {
                'response': f"{entity}: {description}",
                'confidence': response_data.get('confidence', 1.0)
            }
        
        elif response_type == 'facilities':
            entity = response_data.get('entity', 'Unknown')
            facilities = response_data.get('facilities', [])
            description = response_data.get('description', '')
            
            if not facilities:
                response_text = f"No facilities information available for {entity}."
            else:
                facilities_list = ", ".join(facilities)
                response_text = f"{entity} has the following facilities: {facilities_list}."
            
            if description:
                response_text += f"\n{description}"
            
            return {
                'response': response_text,
                'confidence': response_data.get('confidence', 1.0)
            }
        
        elif response_type == 'fallback':
            message = response_data.get('message', "I'm not sure how to help with that.")
            suggestions = response_data.get('suggestions', [])
            
            response_text = message
            if suggestions:
                suggestions_text = "\n".join([f"- {q}" for q in suggestions])
                response_text += f"\n\nYou might want to try:\n{suggestions_text}"
            
            return {
                'response': response_text,
                'confidence': 0.0
            }
        
        elif response_type == 'error':
            return {
                'response': response_data.get('message', "Sorry, I couldn't process your request."),
                'confidence': 0.0
            }
        
        # Default full info response
        info = response_data.get('info', {})
        if not info:
            return {
                'response': "Sorry, I don't have any information about that.",
                'confidence': 0.0
            }
        
        formatted_info = []
        for k, v in info.items():
            if k != 'id' and v is not None:
                if isinstance(v, list):
                    formatted_info.append(f"{k}: {', '.join(map(str, v))}")
                else:
                    formatted_info.append(f"{k}: {v}")
        
        if not formatted_info:
            return {
                'response': f"No detailed information available for {info.get('id', 'this item')}.",
                'confidence': 0.0
            }
        
        return {
            'response': f"Here's what I know about {info.get('id', 'this')}:\n" + 
                       "\n".join(formatted_info),
            'confidence': response_data.get('confidence', 1.0)
        }
        
    except Exception as e:
        logger.error("Error formatting response: %s", str(e), exc_info=True)
        return {
            'response': "Sorry, there was an error formatting the response.",
            'confidence': 0.0
        }

# Initialize the chatbot
chatbot = SimpleChatbot()

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            logger.error("Invalid request data: %s", data)
            return jsonify({
                'error': 'No message provided',
                'response': 'Please provide a message to process.'
            }), 400

        user_message = data['message']
        logger.info("Processing message: %s", user_message)

        # Process the message
        try:
            response_data = chatbot.process_query(user_message)
            logger.info("Generated response data: %s", response_data)
            
            formatted_response = format_response(response_data)
            logger.info("Formatted response: %s", formatted_response)
            
            return jsonify(formatted_response)
        except Exception as e:
            logger.error("Error in query processing: %s", str(e), exc_info=True)
            return jsonify({
                'error': 'Processing error',
                'response': f"Error processing query: {str(e)}"
            }), 500

    except Exception as e:
        logger.error("Error in request handling: %s", str(e), exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'response': "I'm sorry, but I encountered an error processing your request. Please try again."
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'ORB AI is running'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 