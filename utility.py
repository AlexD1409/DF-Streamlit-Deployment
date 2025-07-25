from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from typing import List, Dict

# --- Embedding Generation ---
def create_embedding(text: str, embedding_client: AzureOpenAI) -> List[float]:
    response = embedding_client.embeddings.create(input=text, model="azure/text-embedding-3-small")
    return response.data[0].embedding

# --- Vector Search ---
def vector_search(query: str, vector_client: SearchClient, embedding_client: AzureOpenAI, top_k: int = 5) -> List[Dict]:
    embedding = create_embedding(query, embedding_client)
    results = vector_client.search(
        search_text=None,
        vector_queries=[{
            "vector": embedding,
            "k": top_k,
            "fields": "embedding",
            "kind": "vector"
        }]
    )
    return list(results)

# --- Keyword Search ---
def keyword_search(query: str, keyword_client: SearchClient, top_k: int = 50) -> List[Dict]:
    results = keyword_client.search(
        search_text=query,
        top=top_k,
        include_total_count=True
    )
    return list(results)

# --- LLM Completion ---
def basic_llm_response(prompt: str, chat_client: AzureOpenAI) -> str:
    response = chat_client.chat.completions.create(
        model="azure/o4-mini",
        messages=[
            {"role": "system", "content": """You are a joyful whimsical assistant, 
                              who lives life to full and teaches people
                              how to embrace the whimsical and silly 
                              side of life."""},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
