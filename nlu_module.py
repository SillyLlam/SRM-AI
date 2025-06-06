from typing import Dict, List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLUModule:
    def __init__(self):
        """Initialize the NLU module with necessary models and resources."""
        # Initialize sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('averaged_perceptron_tagger')
        
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Question type patterns
        self.question_patterns = {
            'factual': [
                r'what (is|are)',
                r'where (is|are)',
                r'when (is|are)',
                r'who (is|are)',
                r'which'
            ],
            'procedural': [
                r'how (to|do|can|should)',
                r'what (steps|process)',
                r'guide',
                r'explain'
            ],
            'comparative': [
                r'compare',
                r'difference between',
                r'vs',
                r'versus',
                r'better',
                r'advantages'
            ],
            'yes_no': [
                r'^(is|are|can|should|do|does|will)',
                r'^(has|have|had)'
            ]
        }
        
        # Intent patterns
        self.intent_patterns = {
            'location': [
                r'where',
                r'location',
                r'address',
                r'directions',
                r'find',
                r'reach'
            ],
            'timing': [
                r'when',
                r'timing',
                r'schedule',
                r'hours',
                r'open',
                r'close'
            ],
            'process': [
                r'how to',
                r'process',
                r'steps',
                r'procedure',
                r'apply'
            ],
            'contact': [
                r'contact',
                r'email',
                r'phone',
                r'reach out',
                r'get in touch'
            ],
            'description': [
                r'what is',
                r'tell me about',
                r'describe',
                r'explain'
            ]
        }

    def preprocess_text(self, text: str) -> str:
        """Preprocess text by cleaning and normalizing."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s?]', ' ', text)
        text = ' '.join(text.split())
        
        return text

    def extract_entities(self, text: str) -> List[str]:
        """Extract potential entities from the text using POS tagging."""
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        
        # Extract noun phrases (potential entities)
        entities = []
        current_entity = []
        
        for word, tag in pos_tags:
            if tag.startswith(('NN', 'NNP', 'NNPS')):
                current_entity.append(word)
            elif current_entity:
                entities.append(' '.join(current_entity))
                current_entity = []
        
        if current_entity:
            entities.append(' '.join(current_entity))
        
        return entities

    def detect_question_type(self, text: str) -> str:
        """Detect the type of question (factual, procedural, comparative, yes_no)."""
        text = text.lower().strip()
        
        for q_type, patterns in self.question_patterns.items():
            if any(re.search(pattern, text) for pattern in patterns):
                return q_type
        
        return 'other'

    def detect_intent(self, text: str) -> str:
        """Detect the intent of the query."""
        text = text.lower().strip()
        
        for intent, patterns in self.intent_patterns.items():
            if any(pattern in text for pattern in patterns):
                return intent
        
        return 'general'

    def is_followup_question(self, text: str, context: Dict) -> bool:
        """Determine if the question is a follow-up to previous conversation."""
        text = text.lower()
        
        # Check for pronouns referring to previous context
        pronouns = ['it', 'this', 'that', 'they', 'these', 'those', 'there']
        has_pronouns = any(f" {p} " in f" {text} " for p in pronouns)
        
        # Check for follow-up phrases
        followup_phrases = [
            'what about',
            'how about',
            'tell me more',
            'and',
            'also',
            'what else',
            'more information'
        ]
        has_followup = any(phrase in text for phrase in followup_phrases)
        
        # Check if the question starts with a conjunction
        starts_with_conjunction = any(text.startswith(conj) for conj in ['and', 'but', 'or', 'so'])
        
        return has_pronouns or has_followup or starts_with_conjunction

    def get_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        try:
            # Encode both texts
            embedding1 = self.model.encode([text1])[0]
            embedding2 = self.model.encode([text2])[0]
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {str(e)}")
            return 0.0

    def analyze_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Perform comprehensive analysis of the user query."""
        try:
            # Preprocess the query
            processed_query = self.preprocess_text(query)
            
            # Extract entities
            entities = self.extract_entities(processed_query)
            
            # Detect question type and intent
            question_type = self.detect_question_type(processed_query)
            intent = self.detect_intent(processed_query)
            
            # Check if it's a follow-up question
            is_followup = self.is_followup_question(processed_query, context or {})
            
            # Prepare the analysis result
            analysis = {
                'original_query': query,
                'processed_query': processed_query,
                'entities': entities,
                'question_type': question_type,
                'intent': intent,
                'is_followup': is_followup,
                'context_used': bool(context)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            return {
                'error': str(e),
                'original_query': query
            }

    def get_query_embedding(self, text: str) -> np.ndarray:
        """Get the embedding vector for a text query."""
        try:
            return self.model.encode([text])[0]
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return np.zeros(384)  # Default embedding size for all-MiniLM-L6-v2

    def find_best_matches(self, query: str, candidates: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """Find the best matching candidates for a query."""
        try:
            # Get query embedding
            query_embedding = self.get_query_embedding(query)
            
            # Get embeddings for all candidates
            candidate_embeddings = self.model.encode(candidates)
            
            # Calculate similarities
            similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]
            
            # Get top-k matches
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            return [(candidates[i], similarities[i]) for i in top_indices]
            
        except Exception as e:
            logger.error(f"Error finding best matches: {str(e)}")
            return [] 