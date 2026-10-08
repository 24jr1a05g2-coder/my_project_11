
# ============================================================
# BEGINNER-FRIENDLY RAG DEMO USING LANGCHAIN + CHROMA
# Input file: python1.txt
#
# Steps demonstrated:
# 1. Document Loading
# 2. Text Splitting
# 3. Embeddings
# 4. Chroma Vector Store
# 5. Similarity Search
# 6. Top-K Retrieval
# 7. Context Construction
# 8. Final RAG Answer
# ============================================================


# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

FILE_NAME = "python1.txt"

# Number of chunks to retrieve
TOP_K = 3

# Local folder where Chroma stores vectors
CHROMA_DIR = "./chroma_db"


# ------------------------------------------------------------
# STEP 1: DOCUMENT LOADING
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 1: DOCUMENT LOADING")
print("=" * 70)

print(f"Loading file: {FILE_NAME}")

# encoding='utf-8' avoids many text-file encoding problems
loader = TextLoader(FILE_NAME, encoding="utf-8")

documents = loader.load()

print("\nDocument loaded successfully!")

print("Number of documents:", len(documents))

print("\n--- Document Content ---")
print(documents[0].page_content)

print("\n--- Document Metadata ---")
print(documents[0].metadata)


# ------------------------------------------------------------
# STEP 2: TEXT SPLITTING
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 2: TEXT SPLITTING")
print("=" * 70)

# chunk_size:
# Maximum approximate size of each piece
#
# chunk_overlap:
# Number of characters repeated between neighboring chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Text has been divided into chunks.")

print("Number of chunks:", len(chunks))

print("\n--- CHUNKS ---")

for i in range(len(chunks)):
    print(f"\nCHUNK {i + 1}")
    print("-" * 50)
    print(chunks[i].page_content)


# ------------------------------------------------------------
# STEP 3: CREATE EMBEDDINGS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3: EMBEDDINGS")
print("=" * 70)

print("Loading embedding model...")

# This model runs locally.
# The first execution may download the model.

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")


# Demonstrate one embedding manually

sample_text = chunks[0].page_content

sample_vector = embeddings.embed_query(sample_text)

print("\nSample text:")
print(sample_text)

print("\nEmbedding vector:")
print(sample_vector[:10])

print("\nFirst 10 numbers shown only.")

print("Complete vector length:", len(sample_vector))


# ------------------------------------------------------------
# STEP 4: CREATE CHROMA VECTOR STORE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 4: CHROMA VECTOR STORE")
print("=" * 70)

print("Creating Chroma vector database...")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="python_rag_collection",
    persist_directory=CHROMA_DIR
)

print("Chroma vector store created successfully!")

print("Stored chunks:", len(chunks))

print("Database location:", CHROMA_DIR)


# ------------------------------------------------------------
# STEP 5: ASK A QUESTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 5: USER QUERY")
print("=" * 70)

# You can change this question.

query = input("\nEnter your question about Python: ")

print("\nUSER QUESTION:")
print(query)


# ------------------------------------------------------------
# STEP 6: SIMILARITY SEARCH
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 6: SIMILARITY SEARCH")
print("=" * 70)

print("Searching Chroma for relevant chunks...")

similar_documents = vector_store.similarity_search(
    query,
    k=TOP_K
)

print("\nSimilarity search completed!")

print("Number of retrieved documents:", len(similar_documents))


# ------------------------------------------------------------
# STEP 7: TOP-K RETRIEVAL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print(f"STEP 7: TOP-{TOP_K} RETRIEVAL")
print("=" * 70)

for i, doc in enumerate(similar_documents):

    print(f"\nTOP RESULT {i + 1}")
    print("-" * 50)

    print("Content:")
    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)


# ------------------------------------------------------------
# STEP 8: CONTEXT CONSTRUCTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 8: CONTEXT CONSTRUCTION")
print("=" * 70)

# Combine the retrieved chunks into one context string.

context = "\n\n".join(
    doc.page_content
    for doc in similar_documents
)

print("Retrieved chunks have been combined.")

print("\n--- FINAL CONTEXT SENT TO LLM ---")
print(context)


# ------------------------------------------------------------
# STEP 9: FINAL RAG ANSWER
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 9: FINAL RAG ANSWER")
print("=" * 70)

# ------------------------------------------------------------
# IMPORTANT:
#
# The following simple answer generator demonstrates the RAG
# flow without requiring an external LLM API.
#
# For a real RAG application, replace this section with
# an LLM such as Gemini, OpenAI, Groq, Ollama, etc.
# ------------------------------------------------------------

print("\nQuestion:")
print(query)

print("\nRetrieved Context:")
print(context)

print("\nFinal RAG Response:")
print("-" * 50)

print(
    "The answer should be generated by an LLM using the "
    "retrieved context shown above."
)

print("\nRAG pipeline completed successfully!")


# ------------------------------------------------------------
# RAG PIPELINE SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RAG PIPELINE COMPLETED")
print("=" * 70)

print("""
python1.txt
     |
     v
Document Loading
     |
     v
Text Splitting
     |
     v
Text Chunks
     |
     v
Embeddings
     |
     v
Chroma Vector Store
     |
     v
User Question
     |
     v
Similarity Search
     |
     v
Top-K Retrieved Chunks
     |
     v
Context Construction
     |
     v
LLM
     |
     v
Final Answer
""")
