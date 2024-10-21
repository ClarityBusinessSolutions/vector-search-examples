from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# Connect to MongoDB
client = MongoClient("MONGODB_URI_GOES_HERE")
db = client["vector_db"]
collection = db["vectors"]

# Load the vector embedding pre-trained model
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

texts = ["The quick brown fox jumps over the lazy dog.",
	"The speedy hazel fox leaps over the resting puppy.", 
         "The hotel boasts a very high standard of comfort."]

# Encode the text as vector embeddings
embeddings = model.encode(texts, convert_to_tensor=True).tolist()

documents = [{"text": text, "embedding": embedding} for text, embedding in zip(texts, embeddings)]

collection.insert_many(documents)
