import json
import time
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# -------------------------
# Enable CORS (IMPORTANT)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Root endpoint (optional but helpful)
# -------------------------
@app.get("/")
def root():
    return {"message": "Semantic Search API is running"}

# -------------------------
# Load Documents
# -------------------------
with open("docs.json", "r", encoding="utf-8") as f:
    DOCUMENTS = json.load(f)

TOTAL_DOCS = len(DOCUMENTS)

# -------------------------
# TF-IDF Vectorization
# -------------------------
corpus = [doc["content"] for doc in DOCUMENTS]
vectorizer = TfidfVectorizer(stop_words="english")
doc_vectors = vectorizer.fit_transform(corpus)

# -------------------------
# Request Model
# -------------------------
class SearchRequest(BaseModel):
    query: str
    k: int = 5
    rerank: bool = True
    rerankK: int = 3

# -------------------------
# Normalize Scores (0–1)
# -------------------------
def normalize(scores):
    min_s = np.min(scores)
    max_s = np.max(scores)

    if max_s == min_s:
        return [0.5 for _ in scores]

    return [(float(s - min_s) / float(max_s - min_s)) for s in scores]

# -------------------------
# Search Endpoint
# -------------------------
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

    # Vectorize query
    query_vector = vectorizer.transform([req.query])

    # Compute cosine similarity
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()

    normalized_scores = normalize(similarities)

    results = []
    for i, score in enumerate(normalized_scores):
        results.append({
            "id": DOCUMENTS[i]["id"],
            "score": float(score),
            "content": DOCUMENTS[i]["content"],
            "metadata": DOCUMENTS[i]["metadata"]
        })

    # Sort descending
    results.sort(key=lambda x: x["score"], reverse=True)

    top_k = results[:req.k]

    reranked = False
    if req.rerank and top_k:
        reranked = True
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

