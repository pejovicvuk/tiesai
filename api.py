from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json

from aiAssistant import ask_question, get_vectorstore

app = Flask(__name__)
CORS(app) 

vectorstore = get_vectorstore()

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
        
        formatted_chat_history = []
        if chat_history:
            for item in chat_history:
                # Normalize keys to lowercase to handle case-insensitivity
                if isinstance(item, dict):
                    normalized_item = {k.lower(): v for k, v in item.items()}
                    
                    # Check for presence and ensure the 'role' value is not None
                    if "role" in normalized_item and normalized_item.get("role") and "content" in normalized_item:
                        formatted_chat_history.append({
                            "role": normalized_item["role"],
                            "content": normalized_item["content"] or ""
                        })
                    else:
                        print(f"API Warning: Skipping invalid chat history item (missing or None role/content): {item}")
        
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