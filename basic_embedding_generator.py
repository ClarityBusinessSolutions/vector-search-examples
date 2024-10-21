from sentence_transformers import SentenceTransformer

# Load the vector embedding pre-trained model
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

text = "The quick brown fox jumps over the lazy dog."

# Encode the text as a vector embedding
embedding = model.encode(text, convert_to_tensor=True).tolist()

print(embedding)
