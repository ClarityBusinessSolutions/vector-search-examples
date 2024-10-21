from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# Connect to MongoDB
client = MongoClient("MONGODB_URI_GOES_HERE")
db = client["vector_db"]
collection = db["vectors"]

# Load the vector embedding pre-trained model
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

query = "What is the level of comfort at the hotel?"

# Encode the query as vector embeddings
query_embedding = model.encode(query, convert_to_tensor=True).tolist()

num_candidates = 1000
top_k = 5

# Create the pipeline to search for top k vectors
pipeline = [
    {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": num_candidates,
            "limit": top_k
        }
    },
    {
        "$project": {
            "text": 1,
            "score": {"$meta": "vectorSearchScore"}
        }
    }
]

# Run the pipeline and fetch the results for the current collection
top_results = list(collection.aggregate(pipeline))

for result in top_results:
    print(f"{result["text"]} \n\t\n Cosine Similarity: {result["score"]}\n")