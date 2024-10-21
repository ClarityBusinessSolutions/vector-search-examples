from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm

# Load the vector embedding pre-trained model
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

texts = ["The quick brown fox jumps over the lazy dog.",
	"The speedy hazel fox leaps over the resting puppy.", 
         "The hotel boasts a very high standard of comfort."]


# Encode the text as vector embeddings
embeddings = model.encode(texts, convert_to_tensor=True).tolist()

dot_product = lambda v1, v2: np.dot(v1, v2)
cosine_similarity = lambda v1, v2: np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
euclidean_distance = lambda v1, v2: np.linalg.norm(np.array(v1) - np.array(v2))

# Compare each unique pair of sentences in the list and print their similarity metrics
for i in range(2):
    for j in range(i+1, 3):
        print(f"Sentences being compared: {texts[i]} and {texts[j]} \n \
              Cosine Similarity: {cosine_similarity(embeddings[i], embeddings[j])} \
              Euclidean Distance: {euclidean_distance(embeddings[i], embeddings[j])} \
              Dot Product: {dot_product(embeddings[i], embeddings[j])}")