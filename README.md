# ORB AI Assistant

An intelligent chatbot for SRMIST that uses semantic search and natural language processing to answer questions about campus locations, facilities, and more.

## Features

- Semantic search using sentence transformers
- Intent classification for different types of queries
- Smart fallback with question suggestions
- Location-aware responses with map links
- Robust error handling and logging

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Download the spaCy model:
```bash
python -m spacy download en_core_web_sm
```

3. Start the server:
```bash
python orb_ai.py
```

The server will start on http://0.0.0.0:5000

## API Endpoints

### POST /chat
Send a message to the chatbot.

Request body:
```json
{
    "message": "Where is Tech Park?"
}
```

Response:
```json
{
    "response": "Tech Park is located at SRM Nagar, Kattankulathur, Chengalpattu District, Tamil Nadu - 603203. You can find it in Kattankulathur Campus.\nHere's a map link: https://maps.app.goo.gl/HvLKqGK8TFE5QWLP6",
    "confidence": 0.92
}
```

### GET /health
Check if the service is running.

Response:
```json
{
    "status": "healthy",
    "message": "ORB AI is running"
}
```

## Example Queries

The chatbot can handle various forms of questions about the same topic:

- "Where is Tech Park?"
- "How do I get to Tech Park?"
- "What is Tech Park?"
- "Tell me about Tech Park"
- "What facilities are in Tech Park?"

If the chatbot isn't sure about a query, it will suggest similar questions that it can answer.

## Architecture

The system uses:
- Sentence Transformers for semantic search
- spaCy for text preprocessing
- Flask for the web API
- Structured knowledge base for campus information

## Contributing

To add new locations or facilities to the knowledge base, update the `knowledge_base` dictionary in `semantic_search.py`. 