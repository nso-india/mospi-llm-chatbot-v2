"""
Qdrant helper functions for v1.7.0
"""
import os
import logging
import requests
from typing import Iterable, List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file\

logger = logging.getLogger(__name__)

# Environment configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "information_embeddings")
QDRANT_DISTANCE = os.getenv("QDRANT_DISTANCE", "Cosine")

# gRPC configuration (for better performance)
QDRANT_HOST = os.getenv("QDRANT_HOST", "13.205.85.151")  # Extract host from URL if not set
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))  # Default gRPC port

logger.info(f"📍 QDRANT_URL: {QDRANT_URL}")
logger.info(f"🔑 QDRANT_API_KEY: {'Set' if QDRANT_API_KEY else 'Not set'}")
logger.info(f"📍 QDRANT_HOST: {QDRANT_HOST}")
logger.info(f"📍 QDRANT_GRPC_PORT: {QDRANT_GRPC_PORT}")


def _get_headers():
    """Get headers with API key if available"""
    headers = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        headers["api-key"] = QDRANT_API_KEY
    return headers


def get_qdrant_client() -> QdrantClient:
    """Initialize Qdrant client (HTTP REST API)"""
    try:
        logger.info(f"🔗 Connecting to Qdrant at {QDRANT_URL}")

        kwargs = {"url": QDRANT_URL}
        if QDRANT_API_KEY:
            kwargs["api_key"] = QDRANT_API_KEY

        client = QdrantClient(**kwargs)

        # Test connection
        collections = client.get_collections()
        logger.info(f"✅ Qdrant connected! Collections: {len(collections.collections)}")

        return client

    except Exception as e:
        logger.error(f"❌ Failed to connect: {e}")
        raise RuntimeError(f"Cannot connect to Qdrant at {QDRANT_URL}") from e


def get_qdrant_client_grpc() -> QdrantClient:
    """
    Initialize Qdrant client with gRPC for better performance.
    
    gRPC provides:
    - Lower latency (~30-50% faster than HTTP)
    - More efficient binary protocol
    - Better for high-frequency queries
    
    Returns:
        QdrantClient configured for gRPC
    """
    try:
        logger.info(f"🔗 Connecting to Qdrant via gRPC at {QDRANT_HOST}:{QDRANT_GRPC_PORT}")

        kwargs = {
            "host": QDRANT_HOST,
            "grpc_port": QDRANT_GRPC_PORT,
            "prefer_grpc": True,
            "https": False,  # Disable SSL/TLS for gRPC
        }
        
        if QDRANT_API_KEY:
            kwargs["api_key"] = QDRANT_API_KEY

        client = QdrantClient(**kwargs)

        # Test connection
        collections = client.get_collections()
        logger.info(f"✅ Qdrant connected via gRPC! Collections: {len(collections.collections)}")

        return client

    except Exception as e:
        logger.error(f"❌ Failed to connect via gRPC: {e}")
        logger.warning(f"⚠️ Falling back to HTTP REST API")
        # Fallback to HTTP if gRPC fails
        return get_qdrant_client()


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance: str = QDRANT_DISTANCE
):
    """Create collection if it doesn't exist"""
    headers = _get_headers()

    # Check if collection exists
    try:
        url = f"{QDRANT_URL}/collections/{collection_name}"
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            logger.info(f"✅ Collection '{collection_name}' already exists")
            return
    except:
        pass

    # Create collection - use PUT with collection_name in URL
    logger.info(f"🔨 Creating collection '{collection_name}'...")

    try:
        # v1.7.0 uses PUT /collections/{collection_name}
        url = f"{QDRANT_URL}/collections/{collection_name}"

        payload = {
            "vectors": {
                "size": vector_size,
                "distance": distance
            }
        }

        logger.info(f"🔍 DEBUG - URL: {url}")
        logger.info(f"🔍 DEBUG - Payload: {payload}")
        logger.info(f"🔍 DEBUG - Headers: {headers}")

        response = requests.put(url, json=payload, headers=headers, timeout=30)

        logger.info(f"🔍 DEBUG - Status Code: {response.status_code}")
        logger.info(f"🔍 DEBUG - Response Text: {response.text}")

        if response.status_code in [200, 201]:
            logger.info(f"✅ Collection created with {vector_size}D vectors")
        else:
            logger.error(f"Response: {response.text}")
            raise Exception(f"Status {response.status_code}: {response.text}")

    except Exception as e:
        logger.error(f"❌ Failed to create collection: {e}")
        raise


def build_qdrant_store(collection_name: str, embedding_model) -> QdrantVectorStore:
    """Build LangChain QdrantVectorStore with proper metadata handling for MOSPI collection structure"""
    try:
        client = get_qdrant_client()

        logger.info("📐 Generating sample embedding...")
        sample_vec = embedding_model.embed_query("test")
        vector_size = len(sample_vec)
        logger.info(f"✅ Embedding dimension: {vector_size}D")

        ensure_collection(client, collection_name, vector_size=vector_size)

        # MOSPI collection structure: metadata fields are stored directly in payload
        # Fields: page_content, file_name, file_url, publish_date, topics, page_number, etc.
        # The issue is that LangChain QdrantVectorStore expects nested metadata structure
        # but MOSPI has flat structure where all fields are at payload root level
        
        try:
            # Try with explicit content key mapping for MOSPI structure
            vectordb = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embedding_model,
                content_payload_key="page_content",  # MOSPI uses 'page_content' for text content
            )
            logger.info(f"✅ QdrantVectorStore ready with MOSPI content mapping")
            
            # Test metadata extraction with a sample query
            try:
                test_results = vectordb.similarity_search("test", k=1)
                if test_results:
                    test_doc = test_results[0]
                    test_meta = test_doc.metadata or {}
                    file_name = test_meta.get("file_name", "unknown")
                    logger.info(f"🔍 Metadata test - file_name: {file_name}")
                    if file_name == "unknown":
                        logger.warning("⚠️ Metadata extraction still not working - file_name is 'unknown'")
                    else:
                        logger.info("✅ Metadata extraction working correctly")
            except Exception as test_error:
                logger.warning(f"⚠️ Metadata test failed: {test_error}")
                
        except Exception as init_error:
            logger.warning(f"⚠️ Enhanced initialization failed, using default: {init_error}")
            # Fallback to default initialization
            vectordb = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embedding_model
            )
            logger.info(f"✅ QdrantVectorStore ready with default configuration")

        return vectordb

    except Exception as e:
        logger.error(f"❌ Failed to build Qdrant store: {e}")
        raise


def add_chunks_qdrant(
    client: QdrantClient,
    collection_name: str,
    chunks: List[Dict[str, Any]],
    embedding_model
):
    """Add chunks to Qdrant"""
    if not chunks:
        logger.warning("⚠️ No chunks to add")
        return

    logger.info(f"📤 Adding {len(chunks)} chunks...")

    try:
        texts = [f"passage: {c['content']}" for c in chunks]
        vectors = embedding_model.embed_documents(texts)

        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            meta = chunk["metadata"]

            points.append(
                qmodels.PointStruct(
                    id=i,
                    vector=vector,
                    payload={
                        "content": chunk["content"],
                        "doc_name": meta.get("doc_name"),
                        "doc_id": meta.get("doc_id"),
                        "chunk_id": meta.get("chunk_id"),
                        "chunk_type": meta.get("chunk_type", "text"),
                        "category": meta.get("category", "information"),
                        "uploaded_at": meta.get("uploaded_at"),
                    }
                )
            )

        client.upsert(collection_name=collection_name, points=points)
        logger.info(f"✅ Added {len(points)} chunks")

    except Exception as e:
        logger.error(f"❌ Failed to add chunks: {e}")
        raise


def list_doc_names_qdrant(client: QdrantClient, collection_name: str, limit: int = 100000) -> List[str]:
    """Get unique document names"""
    try:
        doc_names = set()
        offset = None

        while True:
            scroll_result = client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_payload=["doc_name"],
                with_vectors=False
            )

            points, next_offset = scroll_result

            for point in points:
                if point.payload and "doc_name" in point.payload:
                    doc_names.add(point.payload["doc_name"])

            if next_offset is None:
                break
            offset = next_offset

        return sorted(doc_names)

    except Exception as e:
        logger.error(f"❌ Failed to list docs: {e}")
        return []


def delete_chunks_for_doc_qdrant(client: QdrantClient, collection_name: str, doc_names: Iterable[str]):
    """Delete chunks for documents"""
    if isinstance(doc_names, str):
        doc_names = [doc_names]

    for name in doc_names:
        try:
            client.delete(
                collection_name=collection_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="doc_name",
                                match=qmodels.MatchValue(value=name)
                            )
                        ]
                    )
                )
            )
            logger.info(f"✅ Deleted: {name}")
        except Exception as e:
            logger.error(f"❌ Delete failed: {e}")


def update_doc_metadata_qdrant(
    client: QdrantClient,
    collection_name: str,
    doc_names: List[str],
    metadata: Dict[str, Any]
):
    """Update metadata"""
    for name in doc_names:
        try:
            client.set_payload(
                collection_name=collection_name,
                payload=metadata,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="doc_name",
                                match=qmodels.MatchValue(value=name)
                            )
                        ]
                    )
                )
            )
            logger.info(f"✅ Updated: {name}")
        except Exception as e:
            logger.error(f"❌ Update failed: {e}")


def make_retriever(collection_name: str, embedding_model, search_k: int = 10):
    """Create retriever"""
    vectordb = build_qdrant_store(collection_name, embedding_model)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": search_k})
    return vectordb, retriever
