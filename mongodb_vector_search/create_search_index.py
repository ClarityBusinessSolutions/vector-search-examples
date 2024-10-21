from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

# Connect to MongoDB
client = MongoClient("MONGODB_URI_GOES_HERE")
db = client["vector_db"]
collection = db["vectors"]

index_definition = {
    "fields": [
        {
            "numDimensions": 384,
            "path": "embedding",
            "similarity": "cosine",
            "type": "vector"
        }
    ]
}

# Create a SearchIndexModel with the "vectorSearch" type
vector_search_index = SearchIndexModel(definition=index_definition, name="vector_index", type="vectorSearch")

# Create the vector search index in the collection
collection.create_search_indexes([vector_search_index])