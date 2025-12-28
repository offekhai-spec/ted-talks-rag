import pandas as pd
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
import time

# Load variables from the .env file
load_dotenv()

# 1. Initialize Pinecone connection
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# 2. Initialize OpenAI connection via the course proxy (llmod.ai)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

def get_embedding(text):
    """Generates a vector embedding for the given text using the course model."""
    try:
        response = client.embeddings.create(
            input=text,
            model="RPRTHPB-text-embedding-3-small" # Required course model [cite: 33]
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

# 3. Load Data 
print("Loading CSV...")
df = pd.read_csv("ted_talks_en.csv")
print(f"Starting ingestion of {len(df)} talks...")

# Buffer for batch upsert
vectors_batch = []
batch_size = 100  # Pinecone recommendation for efficiency 

for _, row in df.iterrows():
    transcript = str(row['transcript'])
    title = str(row['title'])
    speaker = str(row['speaker_1'])
    talk_id = str(row['talk_id'])

    # Chunking logic: 4000 chars (~1000 tokens) to stay safe under 2048 token limit [cite: 42, 94]
    chunk_size = 4000
    overlap = 800 # 20% overlap, strictly under the 30% limit [cite: 43, 95]
    
    chunks = [transcript[i:i + chunk_size] for i in range(0, len(transcript), chunk_size - overlap)]

    for i, chunk_text in enumerate(chunks):
        context_text = f"Title: {title}\nSpeaker: {speaker}\nContent: {chunk_text}"
        
        vector = get_embedding(context_text)
        
        if vector:
            record_id = f"{talk_id}_{i}"
            vectors_batch.append({
                "id": record_id,
                "values": vector,
                "metadata": {
                    "talk_id": talk_id,
                    "title": title,
                    "text": context_text
                }
            })
        
        # When batch is full, upload it
        if len(vectors_batch) >= batch_size:
            try:
                index.upsert(vectors=vectors_batch)
                print(f"Successfully uploaded a batch of {len(vectors_batch)} vectors.")
                vectors_batch = [] # Clear the batch
                time.sleep(0.5)    # Brief safety pause to avoid network congestion
            except Exception as e:
                print(f"Upsert failed, retrying in 5 seconds... Error: {e}")
                time.sleep(5)

# Upload any remaining vectors in the final batch
if vectors_batch:
    index.upsert(vectors=vectors_batch)
    print(f"Uploaded final batch of {len(vectors_batch)} vectors.")

print("Ingestion Complete!")