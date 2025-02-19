from typing import List
from tqdm import tqdm
import requests
import json

def openai_embed_batch_query(openai_client, texts: List[str], model: str) -> List[List[float]]:
    return [d.embedding for d in openai_client.embeddings.create(model=model, input = texts).data]


def openai_process_in_batches(openai_client, texts: List[str], model: str, batch_size: int = 100) -> List[List[float]]:
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Processing openai batches"):
        batch = texts[i:i + batch_size]
        batch_embeddings = openai_embed_batch_query(openai_client, batch, model)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def embed_batch_voyage_no_input(voyage_client, texts: List[str]) -> List[List[float]]:
    try:
        response = voyage_client.embed(texts, model="voyage-3-large")
        return response.embeddings
    
    except Exception as e:
        print(f"Error embedding batch: {e}")
        return [[0.0]*1024 for _ in texts]


def process_in_batches_voyage_no_input(voyage_client, texts: List[str], batch_size: int = 100) -> List[List[float]]:
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing Voyage batches"):
        batch = texts[i:i + batch_size]
        batch_embeddings = embed_batch_voyage_no_input(voyage_client, batch)
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings


def jina_embed_batch_no_input(texts: List[str]) -> List[List[float]]:
    try:
        url = "https://api.jina.ai/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer jina_api_key"
        }
        
        data = {
            "model": "jina-embeddings-v3",
            "task": "text-matching",
            "late_chunking": False,
            "dimensions": 1024,
            "embedding_type": "float",
            "input": texts
        }

        response = requests.post(url, headers=headers, json=data)
        response_dict = json.loads(response.text)
        embeddings = [item["embedding"] for item in response_dict["data"]]
        
        return embeddings
    
    except Exception as e:
        print(f"Error embedding batch: {e}")
        return [[0.0]*1024 for _ in texts]
    

def jina_process_in_batches_no_input(texts: List[str], batch_size: int = 50) -> List[List[float]]:
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch = texts[i:i + batch_size]

        batch_embeddings = jina_embed_batch_no_input(batch)
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings

def jina_embed_batch_query(texts: List[str]) -> List[List[float]]:
    try:
        url = "https://api.jina.ai/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer jina_api_key"
        }
        
        data = {
            "model": "jina-embeddings-v3",
            "task": "retrieval.query",
            "late_chunking": False,
            "dimensions": 1024,
            "embedding_type": "float",
            "input": texts
        }

        response = requests.post(url, headers=headers, json=data)
        response_dict = json.loads(response.text)
        embeddings = [item["embedding"] for item in response_dict["data"]]
        
        return embeddings
    
    except Exception as e:
        print(f"Error embedding batch: {e}")
        return [[0.0]*1024 for _ in texts]
    

def jina_embed_batch_corpus(texts: List[str]) -> List[List[float]]:
    try:
        url = "https://api.jina.ai/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer jina_api_key"
        }
        
        data = {
            "model": "jina-embeddings-v3",
            "task": "retrieval.passage",
            "late_chunking": False,
            "dimensions": 1024,
            "embedding_type": "float",
            "input": texts
        }

        response = requests.post(url, headers=headers, json=data)
        response_dict = json.loads(response.text)
        embeddings = [item["embedding"] for item in response_dict["data"]]
        
        return embeddings
    
    except Exception as e:
        print(f"Error embedding batch: {e}")
        return [[0.0]*1024 for _ in texts]
    

def jina_process_in_batches_input_type(is_query: bool, texts: List[str], batch_size: int = 100) -> List[List[float]]:
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch = texts[i:i + batch_size]

        if is_query:
            batch_embeddings = jina_embed_batch_query(batch)
        else:
            batch_embeddings = jina_embed_batch_corpus(batch)

        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings


def embed_batch_voyage_corpus(voyage_client, texts: List[str]) -> List[List[float]]:
    try:
        response = voyage_client.embed(texts, model="voyage-3-large", input_type="document")
        return response.embeddings
    
    except Exception as e:
        print(f"Error embedding batch: {e}")
        return [[0.0]*1024 for _ in texts]


def embed_batch_voyage_query(voyage_client, texts: List[str]) -> List[List[float]]:
    try:
        response = voyage_client.embed(texts, model="voyage-3-large", input_type="query")
        return response.embeddings
    
    except Exception as e:
        print(f"Error embedding batch: {e}")
        return [[0.0]*1024 for _ in texts]
    

def process_in_batches_voyage_input_type(voyage_client, is_query: bool, texts: List[str], batch_size: int = 100) -> List[List[float]]:
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing Voyage batches"):
        batch = texts[i:i + batch_size]

        if is_query:
            batch_embeddings = embed_batch_voyage_query(voyage_client, batch)
        else:
            batch_embeddings = embed_batch_voyage_corpus(voyage_client, batch)

        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings


def minilm_embed_batch(model, texts: List[str]) -> List[List[float]]:
    embeddings = model.encode(texts)
    return embeddings

def minilm_process_in_batches(model,texts: List[str], batch_size: int = 100) -> List[List[float]]:
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch = texts[i:i + batch_size]
        batch_embeddings = minilm_embed_batch(model, batch)
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings