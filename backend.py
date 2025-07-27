from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load dataset
with open("dataset_pesan_simpel.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

questions = [item["input"] for item in dataset]
answers = [item["output"] for item in dataset]

# Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

# FastAPI App
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Retrieval chatbot aktif."}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    message = data.get("message", "")

    if not message.strip():
        return {"reply": "Maaf, pertanyaan kosong."}

    # Transformasi dan cari jawaban
    user_vec = vectorizer.transform([message])
    similarities = cosine_similarity(user_vec, tfidf_matrix)
    max_score = similarities.max()
    best_idx = similarities.argmax()

    if max_score < 0.3:
        return {"reply": "Maaf, saya belum punya jawaban untuk itu."}

    return {"reply": answers[best_idx]}
