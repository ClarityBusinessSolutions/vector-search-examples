from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# Load the vector embedding pre-trained model
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# Set up the MongoDB Community database
client = MongoClient('mongodb://localhost:27017/')
db = client['embeddings_db']
collection = db['policies']

# Example policies for a company handbook
policies = [
    "Refunds are only issued for purchases above $100 and must be requested within 30 days of the transaction.",
    "Employees are eligible for five additional vacation days after completing one full year of service.",
    "Payments over $500 must be made via bank transfer and are due within 15 days of receiving the invoice.",
    "Clients who cancel their subscription within the first 10 days are eligible for a full refund minus a $50 processing fee.",
    "Any purchase exceeding $1,000 qualifies for a discount if paid in full within 7 days of purchase."
]

# Embed the policies
embeddings = model.encode(policies, convert_to_tensor=True).tolist()

# Insert the policies into the database
for policy, embedding in zip(policies, embeddings):
    collection.insert_one({
        "policy_text": policy,
        "embedding": embedding
    })