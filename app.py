import json
import time
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List

app = FastAPI()

# ------------------------
# Load Model (FREE Local Model)
# ------------------------
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!")

# ------------------------
# Load Documents
# ------------------------
with open("docs.json", "r", encoding="utf-8") as f:
    DOCUMENTS = json.load(f)

TOTAL_DOCS = len(DOCUMENTS)

# ------------------------
# Compute Document Embeddings (Once)
# ------------------------
print("Computing document embeddings...")
DOC_EMBEDDINGS = model.encode(
    [doc["content"] for doc in DOCUMENTS],
    convert_to_numpy=True
)
print("Embeddings ready!")

# ------------------------
# Cosine Similarity
# ------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def normalize(scores):
    min_s = min(scores)
    max_s = max(scores)

    if max_s == min_s:
        return [0.5 for _ in scores]

    return [(s - min_s) / (max_s - min_s) for s in scores]

# ------------------------
# Simple Re-ranking (Better similarity scoring)
# ------------------------
def rerank(query, candidates):
    query_embedding = model.encode(query, convert_to_numpy=True)

    scores = []
    for doc in candidates:
        doc_embedding = model.encode(doc["content"], convert_to_numpy=True)
        sim = cosine_similarity(query_embedding, doc_embedding)
        scores.append(sim)

    return normalize(scores)

# ------------------------
# Request Model
# ------------------------
class SearchRequest(BaseModel):
    query: str
    k: int = 5
    rerank: bool = True
    rerankK: int = 3

# ------------------------
# Search Endpoint
# ------------------------
@app.post("/search")
def search(req: SearchRequest):
    start = time.time()

    if not req.query.strip():
        return {
            "results": [],
            "reranked": False,
            "metrics": {
                "latency": 0,
                "totalDocs": TOTAL_DOCS
            }
        }

    # Step 1: Embed query
    query_embedding = model.encode(req.query, convert_to_numpy=True)

    # Step 2: Vector Search
    similarities = [
        cosine_similarity(query_embedding, emb)
        for emb in DOC_EMBEDDINGS
    ]

    similarities = normalize(similarities)

    results = []

    for i, score in enumerate(similarities):
        results.append({
            "id": DOCUMENTS[i]["id"],
            "score": float(score),   # <-- ADD float()
            "content": DOCUMENTS[i]["content"],
            "metadata": DOCUMENTS[i]["metadata"]
})


    results.sort(key=lambda x: x["score"], reverse=True)
    top_k = results[:req.k]

    reranked = False

    # Step 3: Re-ranking
    if req.rerank and top_k:
        reranked = True
        new_scores = rerank(req.query, top_k)

        for i in range(len(top_k)):
            top_k[i]["score"] = float(new_scores[i])


        top_k.sort(key=lambda x: x["score"], reverse=True)
        top_k = top_k[:req.rerankK]

    latency = int((time.time() - start) * 1000)

    return {
        "results": top_k,
        "reranked": reranked,
        "metrics": {
            "latency": latency,
            "totalDocs": TOTAL_DOCS
        }
    }
