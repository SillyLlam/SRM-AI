from flask import Flask, request, jsonify
from enhanced_chatbot import EnhancedChatbot
from response_formatter import ResponseFormatter
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
chatbot = EnhancedChatbot()
formatter = ResponseFormatter()

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests."""
    try:
        start_time = time.time()
        
        # Get the query from the request
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'error': 'No query provided',
                'status': 'error'
            }), 400
        
        query = data['query']
        logger.info(f"Received query: {query}")
        
        # Process the query
        response_data = chatbot.process_query(query)
        logger.info(f"Raw response: {response_data}")
        
        # Format the response
        formatted_response = formatter.format_response(response_data)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare the final response
        response = {
            'response': formatted_response['response'],
            'confidence': formatted_response['confidence'],
            'processing_time': round(processing_time, 3),
            'status': 'success'
        }
        
        # Add suggestions if available
        if 'suggestions' in response_data:
            response['suggestions'] = response_data['suggestions']
        
        logger.info(f"Sending response: {response}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'SRMIST Chatbot is running'
    })

if __name__ == '__main__':
    # Initialize the chatbot
    logger.info("Initializing chatbot...")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False) 