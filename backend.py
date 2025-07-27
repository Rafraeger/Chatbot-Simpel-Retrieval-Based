from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
import torch
import json

# Load dataset
with open("dataset_pesan_simpel.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

questions = [item["input"] for item in dataset]
answers = [item["output"] for item in dataset]

# Load Sentence-BERT model (multilingual, ringan, cepat)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Encode seluruh pertanyaan dataset ke bentuk vektor
question_embeddings = model.encode(questions, convert_to_tensor=True)

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
    return {"message": "Retrieval chatbot dengan SBERT aktif."}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    message = data.get("message", "")

    if not message.strip():
        return {"reply": "Maaf, pertanyaan kosong."}

    # Encode pertanyaan user ke vektor
    user_embedding = model.encode(message, convert_to_tensor=True)

    # Hitung kemiripan cosine dengan dataset
    similarities = util.cos_sim(user_embedding, question_embeddings)
    best_idx = torch.argmax(similarities).item()
    max_score = similarities[0][best_idx].item()

    # Batas minimum relevansi
    if max_score < 0.5:
        return {"reply": "Maaf, saya belum punya jawaban untuk itu."}

    return {"reply": answers[best_idx]}
