from flask import Flask, request, jsonify
from chatbot import Chatbot, ChromaDB  
import pandas as pd
from huggingface_hub import login
import os
import uuid

app = Flask(__name__)

huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
login(huggingface_token)
chatbot = Chatbot("meta-llama/Llama-2-7b-chat-hf", huggingface_token)

chroma_db = ChromaDB("chromadb", "all-MiniLM-L6-v2", "chatbotdb")
df = pd.read_csv('/app/topical_chat.csv')
df = df.drop(['sentiment'], axis=1)

messages = df[df['conversation_id']==1]['message'].tolist()
chroma_db.add_documents(messages, [f"id{i}" for i in range(len(messages))])

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data['message']
    
    chroma_db.add_documents([user_input], [str(uuid.uuid4())])
    query_results = chroma_db.query(query_texts=user_input, n_results=5)
    response = chatbot.get_llama_response(user_input, query_results)
    chroma_db.add_documents([response], [str(uuid.uuid4())])
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
