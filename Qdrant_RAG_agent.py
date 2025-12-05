import os
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

# Try to import BaseAgent - you'll need to define this or import it
try:
    from Trading_agent.orchestrator import BaseAgent

except ImportError:
    # Define a minimal BaseAgent if not available
    class BaseAgent:
        def __init__(self):
            pass
        async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
            raise NotImplementedError

# Check if qdrant_client is available
try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


class QdrantRAGAgent(BaseAgent):
    """
    Complete RAG agent with Qdrant vector database and embedding generation.
    Compatible with MultiAgentOrchestrator.
    """
    
    def __init__(
        self,
        collection_name: str = "trading_documents",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        vector_size: int = 384,  # Default for MiniLM-L6-v2
        distance_metric: str = "cosine",
        n_results: int = 5,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        use_local: bool = True,
        local_path: str = "./qdrant_data",
        device: str = "cpu"
    ):
        """
        Initialize complete RAG agent with embeddings.
        """
        super().__init__()
        
        # Configuration
        self.collection_name = collection_name
        self.n_results = n_results
        self.vector_size = vector_size
        
        # Initialize embedding model first
        self.device = device
        self.embedding_model = self._initialize_embedding_model(embedding_model_name)
        
        # Initialize Qdrant client
        self.client = self._initialize_qdrant_client(
            use_local, qdrant_url, qdrant_api_key, local_path
        )
        
        # Distance metric
        self.distance = self._get_distance_metric(distance_metric)
        
        # Create or get collection
        self._ensure_collection_exists()
        
        # Cache for embeddings
        self.embedding_cache = {}
        
        print(f"✅ QdrantRAGAgent initialized: {collection_name}")
        print(f"   Embedding model: {embedding_model_name}")
        print(f"   Vector size: {self.vector_size}")
        print(f"   Qdrant mode: {'Local' if use_local else 'Cloud'}")
    
    def _initialize_embedding_model(self, model_name: str) -> SentenceTransformer:
        """Initialize the embedding model"""
        try:
            model = SentenceTransformer(model_name, device=self.device)
            
            # Test the model to get actual vector size
            test_embedding = model.encode(["test"])
            actual_size = len(test_embedding[0])
            
            # Update vector size if different
            if self.vector_size != actual_size:
                print(f"⚠️ Warning: Model vector size ({actual_size}) "
                      f"doesn't match config ({self.vector_size}). Updating config.")
                self.vector_size = actual_size
            
            print(f"✅ Embedding model loaded. Vector size: {actual_size}")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model {model_name}: {e}")
    
    def _initialize_qdrant_client(
        self,
        use_local: bool,
        qdrant_url: Optional[str],
        qdrant_api_key: Optional[str],
        local_path: str
    ) -> QdrantClient:
        """Initialize Qdrant client"""
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant_client not installed. "
                "Install with: pip install qdrant-client"
            )
        
        try:
            if use_local:
                # Create directory if it doesn't exist
                os.makedirs(local_path, exist_ok=True)
                print(f"Using local Qdrant at: {local_path}")
                return QdrantClient(path=local_path)
            else:
                # Qdrant Cloud
                if not qdrant_url:
                    qdrant_url = os.getenv("QDRANT_URL") or os.getenv("URL_Qdrant")
                if not qdrant_api_key:
                    qdrant_api_key = os.getenv("QDRANT_API_KEY") or os.getenv("API_KEY_Qdrant")
                
                if not qdrant_url:
                    raise ValueError(
                        "Qdrant URL required for cloud mode. "
                        "Set QDRANT_URL env var or pass qdrant_url parameter."
                    )
                
                print(f"Using Qdrant Cloud: {qdrant_url}")
                if qdrant_api_key:
                    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                else:
                    return QdrantClient(url=qdrant_url)
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Qdrant client: {e}")
    
    def _get_distance_metric(self, metric: str) -> Distance:
        """Convert string to Qdrant Distance enum"""
        metric_lower = metric.lower()
        if metric_lower == "cosine":
            return Distance.COSINE
        elif metric_lower == "euclidean":
            return Distance.EUCLID
        elif metric_lower == "dot":
            return Distance.DOT
        else:
            print(f"⚠️ Unknown distance metric '{metric}', using 'cosine'")
            return Distance.COSINE
    
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                print(f"Creating collection: {self.collection_name}")
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=20000
                    )
                )
                
                # Create payload indices for metadata filtering
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="metadata.category",
                        field_schema=models.PayloadSchemaType.KEYWORD
                    )
                    
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="metadata.source",
                        field_schema=models.PayloadSchemaType.KEYWORD
                    )
                except Exception as e:
                    print(f"⚠️ Could not create payload indices: {e}")
                
                print(f"✅ Collection '{self.collection_name}' created")
            else:
                print(f"📁 Using existing collection: {self.collection_name}")
                
                # Verify vector size matches
                collection_info = self.client.get_collection(self.collection_name)
                config_size = collection_info.config.params.vectors.size
                if config_size != self.vector_size:
                    raise ValueError(
                        f"Collection vector size ({config_size}) "
                        f"doesn't match model size ({self.vector_size})"
                    )
                
        except Exception as e:
            print(f"⚠️ Error ensuring collection exists: {e}")
            raise
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if not text or not text.strip():
            return [0.0] * self.vector_size
        
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        # Generate new embedding
        try:
            embedding = self.embedding_model.encode([text])[0].tolist()
            # Cache it
            self.embedding_cache[text_hash] = embedding
            return embedding
        except Exception as e:
            print(f"⚠️ Error generating embedding: {e}")
            return [0.0] * self.vector_size
    
    def _generate_document_id(self, text: str, metadata: Dict) -> str:
        """Generate unique document ID"""
        # Create ID from text hash and timestamp
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        source = metadata.get("source", "unknown")[:20]  # Limit length
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"doc_{source}_{text_hash}_{timestamp}"
    
    def add_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        upsert: bool = True
    ) -> str:
        """
        Add a single document to the vector database.
        """
        if metadata is None:
            metadata = {}
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Generate embedding
        embedding = self._generate_embedding(text)
        
        # Generate document ID if not provided
        if document_id is None:
            document_id = self._generate_document_id(text, metadata)
        
        # Create point for Qdrant
        point = PointStruct(
            id=document_id,
            vector=embedding,
            payload={
                "text": text,
                "metadata": metadata,
                "added_at": datetime.now().isoformat(),
                "text_length": len(text)
            }
        )
        
        # Upsert to Qdrant
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
                wait=True
            )
            print(f"📄 Added document: {document_id} ({len(text)} chars)")
            return document_id
        except Exception as e:
            print(f"❌ Failed to add document {document_id}: {e}")
            raise
    
    def batch_add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100,
        show_progress: bool = True
    ) -> List[str]:
        """
        Add multiple documents efficiently.
        """
        if not documents:
            return []
        
        document_ids = []
        points = []
        
        total_docs = len(documents)
        
        for i, doc in enumerate(documents):
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            
            if not text or not text.strip():
                continue
            
            # Generate embedding
            embedding = self._generate_embedding(text)
            
            # Generate document ID
            doc_id = self._generate_document_id(text, metadata)
            document_ids.append(doc_id)
            
            # Create point
            point = PointStruct(
                id=doc_id,
                vector=embedding,
                payload={
                    "text": text,
                    "metadata": metadata,
                    "added_at": datetime.now().isoformat(),
                    "text_length": len(text)
                }
            )
            points.append(point)
            
            # Upload batch when size is reached
            if len(points) >= batch_size:
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                        wait=True
                    )
                    points = []
                    
                    if show_progress:
                        print(f"📦 Processed {i+1}/{total_docs} documents...")
                except Exception as e:
                    print(f"❌ Batch upload failed: {e}")
        
        # Upload remaining points
        if points:
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True
                )
            except Exception as e:
                print(f"❌ Final batch upload failed: {e}")
        
        print(f"✅ Added {len(document_ids)} documents to {self.collection_name}")
        return document_ids
    
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main agent method - retrieves relevant documents.
        """
        query = context.get("query", "")
        top_k = context.get("top_k", self.n_results)
        
        # Optional filters from context
        filters = context.get("rag_filters", {})
        
        # If no query, return empty
        if not query or not query.strip():
            return {
                "agent": "RAG_AGENT",
                "docs": [],
                "query": "",
                "error": "No query provided",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Build filter if provided
            qdrant_filter = None
            if filters:
                qdrant_filter = self._build_qdrant_filter(filters)
            
            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True,
                query_filter=qdrant_filter,
                score_threshold=0.3  # Minimum similarity score
            )
            
            # Format results
            docs = []
            for result in search_results:
                payload = result.payload or {}
                metadata = payload.get("metadata", {})
                
                docs.append({
                    "id": result.id,
                    "text": payload.get("text", ""),
                    "metadata": metadata,
                    "score": float(result.score),
                    "source": metadata.get("source", "unknown"),
                    "category": metadata.get("category", "general"),
                    "date": metadata.get("date", ""),
                    "added_at": payload.get("added_at", "")
                })
            
            return {
                "agent": "RAG_AGENT",
                "docs": docs,
                "query": query,
                "num_results": len(docs),
                "collection": self.collection_name,
                "top_scores": [doc["score"] for doc in docs[:3]] if docs else [],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ RAG search error: {e}")
            return {
                "agent": "RAG_AGENT",
                "docs": [],
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _build_qdrant_filter(self, filters: Dict[str, Any]) -> Optional[qmodels.Filter]:
        """Build Qdrant filter from filter dict"""
        if not filters:
            return None
        
        conditions = []
        
        # Category filter
        if "category" in filters:
            category = filters["category"]
            conditions.append(
                qmodels.FieldCondition(
                    key="metadata.category",
                    match=qmodels.MatchValue(value=category)
                )
            )
        
        # Source filter
        if "source" in filters:
            source = filters["source"]
            conditions.append(
                qmodels.FieldCondition(
                    key="metadata.source",
                    match=qmodels.MatchValue(value=source)
                )
            )
        
        # Date range filter
        if "date_from" in filters or "date_to" in filters:
            range_dict = {}
            if "date_from" in filters:
                range_dict["gte"] = filters["date_from"]
            if "date_to" in filters:
                range_dict["lte"] = filters["date_to"]
            
            if range_dict:
                conditions.append(
                    qmodels.FieldCondition(
                        key="metadata.date",
                        range=qmodels.Range(**range_dict)
                    )
                )
        
        if not conditions:
            return None
        
        return qmodels.Filter(must=conditions)
    
    def similarity_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Synchronous search for direct use.
        """
        if not query or not query.strip():
            return []
        
        top_k = top_k or self.n_results
        
        try:
            # Generate embedding
            query_embedding = self._generate_embedding(query)
            
            # Build filter
            qdrant_filter = self._build_qdrant_filter(filters) if filters else None
            
            # Search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True,
                query_filter=qdrant_filter
            )
            
            # Format results
            return [
                {
                    "id": result.id,
                    "text": result.payload.get("text", "") if result.payload else "",
                    "metadata": result.payload.get("metadata", {}) if result.payload else {},
                    "score": float(result.score),
                    "source": result.payload.get("metadata", {}).get("source", "") if result.payload else ""
                }
                for result in search_results
            ]
        except Exception as e:
            print(f"❌ Similarity search error: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            # Count points
            count_result = self.client.count(
                collection_name=self.collection_name,
                exact=True
            )
            
            return {
                "collection_name": self.collection_name,
                "status": collection_info.status,
                "vectors_count": collection_info.vectors_count,
                "points_count": count_result.count,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": str(collection_info.config.params.vectors.distance)
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    def load_sample_financial_data(self):
        """Load sample financial documents for testing"""
        sample_docs = [
            {
                "text": "Federal Reserve maintains interest rates steady at 5.25-5.50% range, citing persistent inflation concerns.",
                "metadata": {
                    "source": "FOMC Statement",
                    "date": "2024-01-31",
                    "category": "monetary_policy",
                    "impact": "high",
                    "region": "US"
                }
            },
            {
                "text": "CPI inflation rises to 3.4% year-over-year, exceeding market expectations of 3.2%.",
                "metadata": {
                    "source": "BLS Report",
                    "date": "2024-01-11",
                    "category": "inflation",
                    "impact": "high",
                    "region": "US"
                }
            }
        ]
        
        print("📥 Loading sample financial documents...")
        doc_ids = self.batch_add_documents(sample_docs, show_progress=False)
        print(f"✅ Loaded {len(doc_ids)} sample documents")
        return doc_ids
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.embedding_cache.clear()
        print("🧹 Embedding cache cleared")
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"🗑️ Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"❌ Failed to delete collection: {e}")