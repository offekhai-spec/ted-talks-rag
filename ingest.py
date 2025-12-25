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
            model="RPRTHPB-text-embedding-3-small" # Required course model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

# 3. Load Data (Limited to 20 talks for budget safety)
print("Loading CSV...")
df = pd.read_csv("ted_talks_en.csv").head(20)

print(f"Starting ingestion of {len(df)} talks...")

for _, row in df.iterrows():
    transcript = str(row['transcript'])
    title = str(row['title'])
    speaker = str(row['speaker_1'])
    talk_id = str(row['talk_id'])

    # Chunking logic: ~4000 characters (approx. 1000 tokens)
    # Maximum allowed overlap is 30%; we are using 20% (800 chars)
    chunk_size = 4000
    overlap = 800
    
    chunks = [transcript[i:i + chunk_size] for i in range(0, len(transcript), chunk_size - overlap)]

    for i, chunk_text in enumerate(chunks):
        # Context Injection: Adding metadata to the text before embedding
        context_text = f"Title: {title}\nSpeaker: {speaker}\nContent: {chunk_text}"
        
        # Create the vector
        vector = get_embedding(context_text)
        
        if vector:
            # Upsert to Pinecone
            record_id = f"{talk_id}_{i}"
            index.upsert(vectors=[{
                "id": record_id,
                "values": vector,
                "metadata": {
                    "talk_id": talk_id,
                    "title": title,
                    "text": context_text # Stored to build the prompt later
                }
            }])
            print(f"Uploaded chunk {i} for talk: {title}")
        
        # Brief delay to avoid rate limiting
        time.sleep(0.2)

print("Ingestion Complete!")