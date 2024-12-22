import openai  # Ensure you have installed the `openai` package
from flask import Flask, render_template, jsonify, request, session
import pandas as pd
from openai import AzureOpenAI
from datetime import datetime
import tiktoken
import re
import secrets
import pinecone  # Pinecone for vector database
from pinecone import Pinecone, ServerlessSpec

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Enter your Azure API Key, API Version and Azure Endpoint
# Embedding is done using Azure 
client = AzureOpenAI(
    api_key="",
    api_version="",
    azure_endpoint=""
)

# Enter Pinecone API Key
pc = Pinecone(api_key="")

index_name = ""
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
        pc.create_index(
            name=index_name, 
            dimension=3072, 
            metric='cosine',
            
        )
index_model = pc.Index(index_name)
print(f"Index '{index_name}' is fetched.")

@app.route('/')
def home():
   session.clear()
   return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask_route():
    data = request.get_json()       # Gets the JSON data sent in the request body.
    user_query = data.get('query')  # Extracts the query key from the JSON data.

    response_message = ask(user_query, token_budget=4096 - 100, print_message=False)
    return jsonify({"response": response_message})


# Cleans text by removing HTML tags and extra whitespace.
def clean_text(text):
    cleaned_text = re.sub(r'<.*?>', '', text)
    cleaned_text = re.sub(r'[\t\r\n]+', '', cleaned_text)
    return cleaned_text

def generate_embeddings(text, model="text-embedding-3-large-model"):
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def cosine_similarity(a, b):
    # np.dot(a, b): Computes the dot product between the two vectors.
    # np.linalg.norm(a): Computes the Euclidean norm (magnitude) of vector a.
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


import ast  # for converting embeddings saved as strings back to arrays
import numpy as np

def strings_ranked_by_relatedness(query: str, top_n: int = 100):
    

    """Search Pinecone for similar embeddings based on the query"""
    query_embedding = generate_embeddings(query)
    
    # Upsert the query embedding into Pinecone
    # We need a unique ID for the vector (e.g., a UUID or simple incrementing counter)
    vector_id = str(hash(query))  # Simple hash to generate a unique ID for the query
    metadata = {"text": query}
    
    # Upsert the vector into Pinecone
    index_model.upsert(vectors=[(vector_id, query_embedding, metadata)])

    # Query Pinecone for similar embeddings
    results = index_model.query(
        vector=query_embedding,
        top_k=top_n,
        include_metadata=True
    )

    # Extract text and scores
    strings = [item["metadata"]["text"] for item in results["matches"]]
    relatednesses = [item["score"] for item in results["matches"]]

    return strings, relatednesses

#  Returns the number of tokens in a string based on the model being used (e.g., GPT-4).
def num_tokens(text: str, model: str = "gpt-4") -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def query_message(
    query: str,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, _ = strings_ranked_by_relatedness(query)
    introduction = 'You are a customer assistant that answers questions or give information about text entered by the user from the given data. The Characters before the fisrt space are the Campaign Ids.'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nConcat:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask(
    query: str,
    model: str = "gpt-4",
    token_budget: int = 4096 - 100,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You are a customer assistant that answers based on the questions that are ask."},
        {"role": "user", "content": message}, 
    ]
    response = client.chat.completions.create(
        model="gpt-4",  # Directly pass model here instead of in query_message
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content.strip()

     # Split the response into meaningful paragraphs
    formatted_response = response_message.split('\n\n')

    return formatted_response

if __name__ == '__main__':  
    app.run(debug=True)
