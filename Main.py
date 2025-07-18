
from fastapi import FastAPI, Query
from supabase_client import supabase  # You should configure this module separately
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Lightweight model
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to VeloxG Search API with NLP"}

@app.get("/search")
def search(q: str = Query(..., min_length=1)):
    # Step 1: Fetch all data from Supabase
    result = supabase.from_("veloxg").select("*").execute()
    if result.error:
        return {"error": str(result.error)}

    documents = result.data

    # Step 2: Run semantic similarity
    query_embedding = model.encode(q, convert_to_tensor=True)
    matches = []

    for item in documents:
        combined_text = f"{item.get('title', '')} {item.get('description', '')}"
        doc_embedding = model.encode(combined_text, convert_to_tensor=True)
        score = float(util.pytorch_cos_sim(query_embedding, doc_embedding)[0][0])
        
        if score > 0.4:  # Only return relevant ones
            item["similarity_score"] = round(score * 100, 2)  # percent
            matches.append(item)

    # Step 3: Sort by similarity
    matches = sorted(matches, key=lambda x: x["similarity_score"], reverse=True)

    return {"results": matches[:20]}  # Limit to top 20

@app.post("/add")
def add_link(title: str, url: str, favicon: str = None, description: str = None):
    data = {
        "title": title,
        "url": url,
        "favicon": favicon,
        "description": description,
        "timestamp": datetime.utcnow().isoformat()
    }
    response = supabase.from_("veloxg").insert(data).execute()
    if response.error:
        return {"error": str(response.error)}
    return {"message": "Link added successfully", "data": response.data}
