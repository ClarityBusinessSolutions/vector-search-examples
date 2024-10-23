# Vector Search Code Examples
**Blog Post URL**: *TBD*

---

## Example One
**File:** `basic_embedding_generator.py`  
**Purpose:** Using the Sentence Transformers model `multi-qa-MiniLM-L6-cos-v1`, this code takes an example sentence and converts the text into a **384-dimensional** vector embedding.

---

## Example Two
**File:** `metrics_comparison.py`  
**Purpose:** This code utilizes the Sentence Transformers model `multi-qa-MiniLM-L6-cos-v1` to encode a list of sentences into **384-dimensional** vector embeddings. It then calculates and compares the *cosine similarity, Euclidean distance, and dot product* between each unique pair of sentences, providing insights into their semantic relationships.

---

## Example Three
### Module: `mongodb_vector_search`
**File:** `insert_vectors.py`  
**Purpose:** This code connects to a *MongoDB* database and utilizes the Sentence Transformers model `multi-qa-MiniLM-L6-cos-v1` to encode a list of sentences into **384-dimensional** vector embeddings. It then inserts these embeddings, along with their corresponding texts, into a specified *MongoDB* collection for later retrieval.

**File:** `create_search_index.py`  
**Purpose:** This code establishes a vector search index within a *MongoDB* collection, defining the search configurations for **384-dimensional** vector embeddings. By creating a `SearchIndexModel` with cosine similarity, it allows the collection to efficiently search the vector space.

**File:** `search_vectors.py`  
**Purpose:** This code encodes a user query into a **384-dimensional** vector embedding using the Sentence Transformers model `multi-qa-MiniLM-L6-cos-v1`. It then executes a vector search pipeline on the *MongoDB* collection to retrieve the top five most similar documents based on the query, displaying the results along with their cosine similarity scores.

---

## Example Four
### Module: `retrieval_augmented_generation`
**File:** `insert_vectors.py`  
**Purpose:** This code connects to a local *MongoDB* database and utilizes the Sentence Transformers model `multi-qa-MiniLM-L6-cos-v1` to encode a set of example company policies into **384-dimensional** vector embeddings. It then inserts these embeddings along with the corresponding policy texts into the *MongoDB* collection for future retrieval and analysis.

**File:** `generate_response.py`  
**Purpose:** This code encodes a user query regarding company policies into a **384-dimensional** vector embedding using the Sentence Transformers model `multi-qa-MiniLM-L6-cos-v1`. It retrieves all stored policies from the *MongoDB* database, computes the cosine similarity between the query and each policy embedding, and identifies the most similar policy. The relevant policy context is then used to generate a context-specific response using the `Llama 3.2` model, ensuring data privacy by running the model locally. Alternatively, the code includes commented-out sections for using *Google Gemini*, providing an option for users who may prefer to utilize external API services for response generation.
