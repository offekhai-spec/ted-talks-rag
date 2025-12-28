import os
from fastapi import FastAPI
from pydantic import BaseModel
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Database and LLM Connections
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Connect through the course proxy server
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
async def root():
    """
    Root endpoint to prevent 404 errors on the home page.
    """
    return {"message": "TED Talk RAG API is running. Use /api/prompt for queries or /api/stats for configuration."}

@app.get("/api/stats")
async def get_stats():
    """
    Returns the RAG configuration. 
    Hyperparameters chosen for efficiency and recall balance:
    
    - chunk_size: 1024 tokens (approx 4000 chars) was chosen to balance semantic richness and prompt efficiency. 
      This size is large enough to capture complete ideas and logical paragraphs from TED transcripts 
      without exceeding model context limits.
      
    - overlap_ratio: 0.2 (20%) ensures that no critical information is lost at the boundaries of text splits. 
      This overlap maintains context continuity, allowing the system to preserve semantic meaning 
      even when a topic spans across multiple chunks.
      
    - top_k: 10 was selected to ensure high recall for multi-talk queries. Since a single talk 
      might yield multiple relevant chunks, a higher value of k provides enough diversity to 
      identify and present at least three unique talk titles.
    """
    return {
        "chunk_size": 1024,
        "overlap_ratio": 0.2,
        "top_k": 10
    }

@app.post("/api/prompt")
async def ask_question(request: QuestionRequest):
    """The main RAG endpoint[cite: 63]."""
    
    # 1. Create embedding for the user's question
    q_emb = client.embeddings.create(
        input=request.question,
        model="RPRTHPB-text-embedding-3-small"
    ).data[0].embedding
    
    # 2. Search Pinecone for the Top-k most similar chunks
    results = index.query(vector=q_emb, top_k=10, include_metadata=True)
    
    # Extract text contexts and prepare metadata for the JSON response
    contexts = []
    metadata_list = []
    
    for res in results['matches']:
        contexts.append(res['metadata']['text'])
        metadata_list.append({
            "talk_id": res['metadata']['talk_id'],
            "title": res['metadata']['title'],
            "score": res['score'],
            "chunk": res['metadata']['text'] # Full chunk text for context 
        })

    # 3. Build the Augmented Prompt using the required system instructions 
    combined_context = "\n\n".join(contexts)
    system_prompt = (
        "You are a TED Talk assistant that answers questions strictly and only based on the "
        "TED dataset context provided to you (metadata and transcript passages). "
        "You must not use any external knowledge, the open internet, or information that is not "
        "explicitly contained in the retrieved context. If the answer cannot be determined "
        "from the provided context, respond: 'I don't know based on the provided TED data.' "
        "Always explain your answer using the given context.\n\n"
        f"Context:\n{combined_context}"
    )
    
    # 4. Generate the final answer using the course LLM 
    response = client.chat.completions.create(
        model="RPRTHPB-gpt-5-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.question}
        ]
    )
    
    # 5. Return the response in the mandatory JSON format 
    return {
        "response": response.choices[0].message.content,
        "context": metadata_list,
        "Augmented_prompt": {
            "System": system_prompt,
            "User": request.question
        }
    }