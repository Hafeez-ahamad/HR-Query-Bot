import json
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import openai
import os

# Load your OpenAI API key from environment variable
from openai import OpenAI
import os

# Create OpenAI client

client = OpenAI(api_key="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"
)
# Initialize FastAPI app
app = FastAPI(title="HR Resource Query Chatbot")

# 1️⃣ Load employee dataset
with open("employees.json", "r") as f:
    employees = json.load(f)["employees"]

# 2️⃣ Load embeddings model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 3️⃣ Create text representation for each employee
employee_texts = [
    f"{emp['name']} has skills {', '.join(emp['skills'])} "
    f"with {emp['experience_years']} years of experience. "
    f"Projects: {', '.join(emp['projects'])}. "
    f"Availability: {emp['availability']}."
    for emp in employees
]

# 4️⃣ Create embeddings and FAISS index
employee_embeddings = embedder.encode(employee_texts, convert_to_tensor=False)
dimension = employee_embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(employee_embeddings))

# Request body model
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/chat")
def chat_with_hr_bot(request: QueryRequest):
    """Search employees and return AI-generated HR answer."""
    try:
        # Embed query
        query_embedding = embedder.encode([request.query], convert_to_tensor=False)
        distances, indices = index.search(np.array(query_embedding), request.top_k)

        # Get matched employees
        matched = [employees[i] for i in indices[0]]

        # Create context
        context = "\n".join(
            [f"{emp['name']} - Skills: {', '.join(emp['skills'])}, "
             f"Experience: {emp['experience_years']} years, "
             f"Projects: {', '.join(emp['projects'])}, "
             f"Availability: {emp['availability']}"
             for emp in matched]
        )

        # Call GPT
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
        {"role": "system", "content": "You are an HR assistant that recommends employees for projects."},
        {"role": "user", "content": f"User query: {request.query}\n\nEmployee data:\n{context}"}
             ]
        )

        answer = answer = response.choices[0].message.content
        return {"answer": answer, "matches": matched}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/employees/search")
def search_employees(skill: str = None, available: bool = None):
    """Search employees by skill and/or availability."""
    results = employees
    if skill:
        results = [emp for emp in results if skill.lower() in [s.lower() for s in emp["skills"]]]
    if available is not None:
        avail_str = "available" if available else "unavailable"
        results = [emp for emp in results if emp["availability"].lower() == avail_str]
    return results
