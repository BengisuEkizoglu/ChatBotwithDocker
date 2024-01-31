# -*- coding: utf-8 -*-

from langchain.docstore.document import Document
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import gradio as gr
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
from transformers import pipeline, AutoTokenizer
import torch


class Chatbot:
    SYSTEM_PROMPT = """<s>[INST] <<SYS>>
    You are a helpful bot. Your answers are clear and concise.
    <</SYS>>

    """
    
    def __init__(self, model_name, token, memory_limit=5):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
        self.pipeline = pipeline("text-generation", model=model_name, torch_dtype=torch.float16, device_map="auto", use_auth_token=token)
        self.memory_limit = memory_limit

    def format_message(message: str, history: list, memory_limit: int = 5) -> str:
        """
        Formats the message and history for the Llama model.

        Parameters:
            message (str): Current message to send.
            history (list): Past conversation history.
            memory_limit (int): Limit on how many past interactions to consider.

        Returns:
            str: Formatted message string
        """
        if len(history) > memory_limit:
            history = history[-memory_limit:]

        if len(history) == 0:
            return SYSTEM_PROMPT + f"{message} [/INST]"

        formatted_message = SYSTEM_PROMPT + f"{history[0][0]} [/INST] {history[0][1]} </s>"

        for user_msg, model_answer in history[1:]:
            formatted_message += f"<s>[INST] {user_msg} [/INST] {model_answer} </s>"

        formatted_message += f"<s>[INST] {message} [/INST]"

        return formatted_message

    def get_llama_response(message: str, history: list) -> str:
        """
        Generates a conversational response from the Llama model.

        Parameters:
            message (str): User's input message.
            history (list): Past conversation history.

        Returns:
            str: Generated response from the Llama model.
        """
        query = format_message(message, history)
        print(query)
        response = ""

        sequences = llama_pipeline(
            query,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=1024,
        )

        generated_text = sequences[0]['generated_text']
        response = generated_text[len(query):]  

        print("Chatbot:", response.strip())
        return response.strip()
    
class ChromaDB:
    def __init__(self, data_path, embed_model, collection_name):
        self.client = chromadb.PersistentClient(path=data_path)
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model)
        self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=embedding_func, metadata={"hnsw:space": "cosine"})

    def add_documents(self, documents, ids):
        self.collection.add(documents=documents, ids=ids)

    def query(self, query_text, n_results=5):
        return self.collection.query(query_texts=[query_text], n_results=n_results)
