import unittest
from enhanced_chatbot import EnhancedChatbot
import json
import logging

class TestEnhancedChatbot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.chatbot = EnhancedChatbot()
        cls.session_id = cls.chatbot.start_conversation()

    def test_basic_query(self):
        """Test basic query handling."""
        query = "Where is the Tech Park?"
        response = self.chatbot.process_query(query, self.session_id)
        
        self.assertIsNotNone(response)
        self.assertIn('response', response)
        self.assertIn('confidence', response)
        self.assertTrue(response['confidence'] > 0.6)
        self.assertIn('Tech Park', response['response'])

    def test_followup_query(self):
        """Test follow-up query handling."""
        # First query
        query1 = "Tell me about the Central Library"
        response1 = self.chatbot.process_query(query1, self.session_id)
        
        # Follow-up query
        query2 = "What are its timings?"
        response2 = self.chatbot.process_query(query2, self.session_id)
        
        self.assertIsNotNone(response2)
        self.assertIn('response', response2)
        self.assertTrue(response2['confidence'] > 0.6)
        self.assertIn('timing', response2['response'].lower())

    def test_unknown_query(self):
        """Test handling of unknown queries."""
        query = "What is the meaning of life?"
        response = self.chatbot.process_query(query, self.session_id)
        
        self.assertIsNotNone(response)
        self.assertIn('response', response)
        self.assertTrue(response['confidence'] < 0.6)
        self.assertIn('suggestions', response)

    def test_entity_recognition(self):
        """Test entity recognition in queries."""
        queries = [
            "Where is Tech Park located?",
            "Tell me about the Central Library",
            "What are the hostel facilities?"
        ]
        
        for query in queries:
            response = self.chatbot.process_query(query, self.session_id)
            self.assertIsNotNone(response)
            self.assertIn('response', response)
            self.assertTrue(response['confidence'] > 0.6)

    def test_conversation_context(self):
        """Test conversation context management."""
        # Initial query
        query1 = "Tell me about Tech Park"
        response1 = self.chatbot.process_query(query1, self.session_id)
        
        # Follow-up queries
        queries = [
            "What facilities does it have?",
            "Where is it located?",
            "How can I reach there?"
        ]
        
        for query in queries:
            response = self.chatbot.process_query(query, self.session_id)
            self.assertIsNotNone(response)
            self.assertIn('response', response)
            self.assertTrue(response['confidence'] > 0.5)

    def test_intent_detection(self):
        """Test intent detection for different query types."""
        test_cases = [
            ("Where is the Tech Park?", "location"),
            ("What facilities does the library have?", "facilities"),
            ("How do I contact the admissions office?", "contact"),
            ("Tell me about the hostel", "description")
        ]
        
        for query, expected_intent in test_cases:
            response = self.chatbot.process_query(query, self.session_id)
            self.assertIn('response', response)
            self.assertTrue(response['confidence'] > 0.6)

    def test_fallback_mechanism(self):
        """Test fallback mechanism for unclear queries."""
        unclear_queries = [
            "What about that thing?",
            "Can you help me?",
            "I need information"
        ]
        
        for query in unclear_queries:
            response = self.chatbot.process_query(query, self.session_id)
            self.assertIn('suggestions', response)
            self.assertTrue(len(response['suggestions']) > 0)

    def test_conversation_summary(self):
        """Test conversation summary functionality."""
        # Make a series of queries
        queries = [
            "Where is Tech Park?",
            "What facilities does it have?",
            "Tell me about the Central Library"
        ]
        
        for query in queries:
            self.chatbot.process_query(query, self.session_id)
        
        # Get conversation summary
        summary = self.chatbot.get_conversation_summary(self.session_id)
        
        self.assertIsNotNone(summary)
        self.assertIn('num_interactions', summary)
        self.assertEqual(summary['num_interactions'], len(queries))
        self.assertIn('mentioned_entities', summary)

    def test_error_handling(self):
        """Test error handling capabilities."""
        # Test with empty query
        response = self.chatbot.process_query("", self.session_id)
        self.assertIn('error', response)
        
        # Test with None query
        response = self.chatbot.process_query(None, self.session_id)
        self.assertIn('error', response)

if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Run the tests
    unittest.main() 