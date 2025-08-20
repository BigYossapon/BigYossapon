import os
import time
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from elasticsearch import Elasticsearch, helpers
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.schema import Document

# config from env
ES_URL = os.getenv("ES_URL", "http://elasticsearch:9200")
ES_INDEX = os.getenv("ES_INDEX", "rag_documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# settings
BULK_BATCH_SIZE = int(os.getenv("BULK_BATCH_SIZE", "500"))  # batch size for bulk indexing
ES_CONNECT_RETRIES = int(os.getenv("ES_CONNECT_RETRIES", "10"))
ES_CONNECT_DELAY = float(os.getenv("ES_CONNECT_DELAY", "2.0"))

app = FastAPI(title="RAG FastAPI (optimized for large scale)")

# ---- helper funcs ----
def wait_for_es(es_client: Elasticsearch, retries=ES_CONNECT_RETRIES, delay=ES_CONNECT_DELAY):
    for i in range(retries):
        try:
            if es_client.ping():
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False

def clean_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not metadata:
        return {}
    return {k: v for k, v in metadata.items() if v is not None}


def create_index_if_not_exists(es_client: Elasticsearch, index_name: str, dim: int = 768):
    if es_client.indices.exists(index=index_name):
        return

    mapping_template = {
        "mappings": {
            "properties": {
                "metadata": {
                    "type": "object",
                    "enabled": True,   # เก็บใน _source
                    "dynamic": False   # ไม่สร้าง field ย่อย
                }
            }
        }
    }

    # Check Elasticsearch version and adjust mapping accordingly
    try:
        info = es_client.info()
        version = info['version']['number']
        major_version = int(version.split('.')[0])

        if major_version >= 8:
            # Elasticsearch 8.x specific changes
            mapping_template["mappings"]["properties"]["vector"] = {
                "type": "dense_vector",
                "dims": dim
            }
        elif major_version == 7:
            # Elasticsearch 7.x specific changes
            mapping_template["mappings"]["properties"]["vector"] = {
                "type": "knn_vector",
                "dimension": dim,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib"
                }
            }
        else:
            raise RuntimeError(f"Unsupported Elasticsearch version: {version}")

        es_client.indices.create(index=index_name, body=mapping_template)
    except Exception as e:
        raise RuntimeError(f"Failed to create index: {e}")


# ---- init clients (wait for ES) ----
es_client = Elasticsearch(ES_URL)
if not wait_for_es(es_client):
    # Fail fast — in production you might want to loop longer or use healthchecks in compose
    raise RuntimeError(f"Elasticsearch not reachable at {ES_URL}")

# ensure index exists with appropriate mapping (dimension depends on your embedding model)
# default embed dim for intfloat/multilingual-e5-base is 768 — change if different
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))


# # Embedding model (HuggingFace)
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ---- API models ----
class DocumentInputIndex(BaseModel):
    doc_id: Optional[str] = None
    page_content: str
    metadata: Optional[Dict[str, Any]] = None
    index: str

class DocumentInput(BaseModel):
    doc_id: Optional[str] = None
    page_content: str
    metadata: Optional[Dict[str, Any]] = None


class BulkDocumentInput(BaseModel):
    documents: List[DocumentInput]
    index: str

class QueryInput(BaseModel):
    query: str
    k: Optional[int] = 5
    index: str
    model: str
    filter_metadata: Optional[Dict[str, Any]] = None  # optional pre-filter

# ---- endpoints ----
@app.post("/add_document")
def add_document(doc: DocumentInputIndex):
    
    vector_store = ElasticsearchStore(
    es_url=ES_URL,
    index_name=doc.index,
    embedding=embedding_model
    )
    create_index_if_not_exists(es_client, doc.index, dim=EMBED_DIM)
    # skip if exists

    if doc.doc_id and es_client.exists(index=doc.index, id=doc.doc_id):
        return {"status": "skipped", "reason": "exists"}
    
    cleaned_meta = clean_metadata(doc.metadata)
    lc_doc = Document(page_content=doc.page_content, metadata=cleaned_meta)

    # add single (LangChain will compute embeddings and push to ES)
    vector_store.add_documents([lc_doc], ids=[doc.doc_id] if doc.doc_id else None)
    return {"status": "ok"}

@app.post("/add_documents_bulk")
def add_documents_bulk(payload: BulkDocumentInput):
    vector_store = ElasticsearchStore(
    es_url=ES_URL,
    index_name=payload.index,
    embedding=embedding_model
    )
    create_index_if_not_exists(es_client, payload.index, dim=EMBED_DIM)
    """
    Bulk add many documents (in batches).
    This uses LangChain's add_documents in batches so embeddings are computed per-batch.
    For extreme scale consider computing embeddings separately and using es_client.bulk.
    """
    docs = payload.documents
    n = len(docs)
    if n == 0:
        return {"status": "no_docs"}

    added = 0
    # batch convert and insert
    batch: List[Document] = []
    ids_batch: List[str] = []
    for i, d in enumerate(docs, start=1):
        if d.doc_id and es_client.exists(index=payload.index, id=d.doc_id):
            continue
        cleaned_meta = clean_metadata(d.metadata)
        batch.append(Document(page_content=d.page_content, metadata=cleaned_meta))
        ids_batch.append(d.doc_id if d.doc_id else None)

        if len(batch) >= BULK_BATCH_SIZE:
            vector_store.add_documents(batch, ids=[i for i in ids_batch] if any(ids_batch) else None)
            added += len(batch)
            batch = []
            ids_batch = []

    # final batch
    if batch:
        vector_store.add_documents(batch, ids=[i for i in ids_batch] if any(ids_batch) else None)
        added += len(batch)

    return {"status": "ok", "added": added}

@app.post("/search")
def search_rag(q: QueryInput):
    """
    Search endpoint with optional metadata filter and variable k.
    For large-scale use, prefer pre-filtering (metadata) to narrow search scope.
    """
    # Vector store (LangChain ElasticsearchStore)
    try:
    
        vector_store = ElasticsearchStore(
            es_url=ES_URL,
            index_name=q.index,
            embedding=embedding_model
        )

        # LLM (Ollama)
        llm = Ollama(model=q.model, base_url=OLLAMA_BASE_URL)

        # Retrieval QA chain (you can tune chain_type, k, etc.)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
        )
    except Exception as e:
      print('An exception occurred')
      return {"msg": str(e)}

    # apply k
    k = q.k or 5

    # If metadata filter provided: we can directly call ES to prefilter doc ids then use vector search
    if q.filter_metadata:
        # build a simple term bool filter for ES
        must_clauses = []
        for key, val in q.filter_metadata.items():
            # only include not-null filters
            if val is None:
                continue
            # if value is list -> terms, else term
            if isinstance(val, list):
                must_clauses.append({"terms": {f"metadata.{key}": val}})
            else:
                must_clauses.append({"term": {f"metadata.{key}": {"value": val}}})

        # fetch candidate ids first (limit to some)
        body = {
            "size": 1000,  # adjust
            "_source": False,
            "query": {"bool": {"filter": must_clauses}}
        }
        res = es_client.search(index=ES_INDEX, body=body)
        ids = [hit["_id"] for hit in res["hits"]["hits"]]
        if not ids:
            return {"query": q.query, "results": [], "note": "no docs after filter"}

        # If we have candidate ids, use retriever with filter by ids (LangChain retriever might not support id filter directly)
        # Simpler: retrieve embeddings for query then run ES knn with ids filter (we implement here manually)
        query_embedding = embedding_model.embed_query(q.query)  # returns a vector list

        # craft knn (or script_score) query with ids filter — best-effort: try knn style
        knn_body = {
            "knn": {
                "field": "vector",
                "query_vector": query_embedding,
                "k": k,
                "num_candidates": 100
            },
            "query": {
                "ids": {"values": ids}
            }
        }
        try:
            es_res = es_client.search(index=ES_INDEX, body=knn_body)
            hits = es_res.get("hits", {}).get("hits", [])
            results = [{"id": h["_id"], "score": h["_score"], "source": h.get("_source")} for h in hits]
            return {"query": q.query, "results": results}
        except Exception:
            # fallback: use script_score hybrid
            script_body = {
                "size": k,
                "query": {
                    "script_score": {
                        "query": {"ids": {"values": ids}},
                        "script": {
                            "source": "cosineSimilarity(params.q, 'embedding') + 1.0",
                            "params": {"q": query_embedding}
                        }
                    }
                }
            }
            es_res = es_client.search(index=ES_INDEX, body=script_body)
            hits = es_res.get("hits", {}).get("hits", [])
            results = [{"id": h["_id"], "score": h["_score"], "source": h.get("_source")} for h in hits]
            return {"query": q.query, "results": results}

    # If no pre-filter: use LangChain retriever directly (k)
    retriever.search_kwargs = {"k": k}
    answer = qa.invoke(q.query)
    print(answer["source_documents"])
    # answer = qa.run(q.query)
    # return {"query": q.query, "answer": answer}
    return {"query": q.query, "answer": answer["result"]}

# health
@app.get("/health")
def health():
    return {"es_connected": es_client.ping(), "index": ES_INDEX}