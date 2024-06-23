
"""
IMPORT RELEVANT LIBRARIES
"""

import openai
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer


"""
Let's build the vector database to generate relevant context for the user queries
"""

import pandas as pd
df = pd.read_csv('..//wine-ratings.csv')
df = df[df['variety'].notna()] 
data = df.sample(700).to_dict('records') 


encoder = SentenceTransformer('all-MiniLM-L6-v2') # Model to create embeddings

# create the vector database client
qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance

# Create collection to store wines
qdrant.recreate_collection(
    collection_name="top_wines",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(), # Vector size is defined by used model
        distance=models.Distance.COSINE
    )
)

# vectorize!
qdrant.upsert(
    collection_name="top_wines",
    points=[
        models.PointStruct(
            id=idx,
            vector=encoder.encode(doc["notes"]).tolist(),
            payload=doc,
        ) for idx, doc in enumerate(data) 
    ]
)



"""
Initiate a client to connect with locally running llamafile model - TinyLlama-1.1B-Chat-v1.0.F16.llamafile
"""

client = openai.OpenAI(
    base_url="http://127.0.0.1:8080/v1", # "http://<Your api-server IP>:port"
    api_key="sk-no-key-required"
)



"""
Let's finally create a fastapi based web server to get user user query and send it to 
vector db first to generate rleevant context and then to openai model for reponse generation
"""

app = FastAPI()


class Body(BaseModel):
    query: str


@app.get('/')
def root():
    return RedirectResponse(url='/docs', status_code=301)


@app.post('/ask')
def ask(body: Body):
    """
    Use the query parameter to interact with the llamafile model
    using qdrant Client for Retrieval Augmented Generation.
    """
    search_result = search(body.query)
    chat_bot_response = assistant(body.query, search_result)
    return {'response': chat_bot_response}



def search(query):
    """
    Send the query to qdrant database for search relevant context,
    """

    hits = qdrant.search(
    collection_name="top_wines",
    query_vector=encoder.encode(query).tolist(),
    limit=3
)
    search_results = [hit.payload for hit in hits]

    return search_results


def assistant(query, context):
    completion = client.chat.completions.create(
    model="LLaMA_CPP",
    messages=[
        {"role": "system", "content": "Asisstant is a chatbot that helps you find the best wine for your taste."},
        {"role": "user", "content": query},
        {"role": "assistant", "content": str(context)}
    ]
    )

    print(query)
    print(context)
    return completion.choices[0].message
