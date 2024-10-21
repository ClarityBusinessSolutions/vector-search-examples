import requests
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import ollama

# Load the vector embedding pre-trained model
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# Set up the MongoDB Community database
client = MongoClient('mongodb://localhost:27017/')
db = client['embeddings_db']
collection = db['policies']

query_text = "What is the subscription cancellation policy?"

# Encode the query
query_embedding = model.encode(query_text, convert_to_tensor=True).tolist()

# Fetch all policies and embeddings from MongoDB
policies_data = list(collection.find({}, {"_id": 0, "policy_text": 1, "embedding": 1}))

# Extract embeddings and texts from the database results
embeddings = [policy["embedding"] for policy in policies_data]
policy_texts = [policy["policy_text"] for policy in policies_data]

# Compute cosine similarity
similarities = cosine_similarity([query_embedding], embeddings)[0]

# Find the text of the most similar policy
context = policy_texts[np.argmax(similarities)]

# Define the query with added context
final_query = query_text + " --- with this information from the company handbook: " + context 

# Running the Llama 3.2 model locally for complete data privacy (no external API calls)
response = ollama.generate(model='llama3.2:latest', prompt=final_query)
print(response['response'])

# # To use Google Gemini model instead of Llama 3.2
# url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
# api_key = "GEMINI_API_KEY_GOES_HERE"

# # Define the data payload
# data = {
#     "contents": [
#         {
#             "parts": [
#                 {
#                     "text": final_query
#                 }
#             ]
#         }
#     ]
# }

# response = requests.post(
#     url,
#     headers={"Content-Type": "application/json"},
#     params={"key": api_key},
#     data=json.dumps(data)
# ).json()

# print(response["candidates"][0]["content"]["parts"][0]["text"])
