from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json

# Import your existing functions
# Assuming ask_question and get_vectorstore are from your existing code
from aiAssistant import ask_question, get_vectorstore

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-domain requests

# Load the vector store once at startup
print("Initializing vector store...")
vectorstore = get_vectorstore()
print("Vector store initialized!")

@app.route('/api/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        question = data.get('question')
        chat_history = data.get('chat_history', [])
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        # Convert chat history format if needed
        formatted_chat_history = []
        for item in chat_history:
            formatted_chat_history.append({
                "role": item.get("role"),
                "content": item.get("content")
            })
        
        # Get response from your existing function
        answer, sources, attachment_ids = ask_question(question, formatted_chat_history, vectorstore)
        
        return jsonify({
            "answer": answer,
            "sources": sources,
            "attachment_ids": attachment_ids
        })
    
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)