"""Vector store for semantic search."""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import structlog

logger = structlog.get_logger()


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add(self, id: str, embedding: List[float], metadata: dict) -> None:
        """Add a vector with metadata."""
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float, dict]]:
        """
        Search for similar vectors.
        
        Returns:
            List of (id, similarity_score, metadata) tuples
        """
        pass
    
    @abstractmethod
    def get(self, id: str) -> Optional[Tuple[List[float], dict]]:
        """Get vector and metadata by ID."""
        pass
    
    @abstractmethod
    def delete(self, id: str) -> None:
        """Delete a vector by ID."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all vectors."""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store."""
    
    def __init__(self, dimension: int = 384):
        """Initialize FAISS vector store."""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")
        
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.id_to_index: dict[str, int] = {}
        self.index_to_id: dict[int, str] = {}
        self.metadata_store: dict[str, dict] = {}
        self.next_index = 0
        
        logger.info("FAISSVectorStore initialized", dimension=dimension)
    
    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """Normalize vector for cosine similarity."""
        norm = np.linalg.norm(vec)
        if norm > 0:
            return vec / norm
        return vec
    
    def add(self, id: str, embedding: List[float], metadata: dict) -> None:
        """Add a vector with metadata."""
        if len(embedding) != self.dimension:
            raise ValueError(f"Embedding dimension {len(embedding)} != {self.dimension}")
        
        vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
        vec = self._normalize(vec[0]).reshape(1, -1)
        
        if id in self.id_to_index:
            # Update existing
            idx = self.id_to_index[id]
            self.index.remove_ids(np.array([idx]))
            self.index.add(vec)
            self.index_to_id[idx] = id
        else:
            # Add new
            idx = self.next_index
            self.index.add(vec)
            self.id_to_index[id] = idx
            self.index_to_id[idx] = id
            self.next_index += 1
        
        self.metadata_store[id] = metadata
        logger.debug("Added vector", id=id[:8], dimension=self.dimension)
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float, dict]]:
        """Search for similar vectors."""
        if len(query_embedding) != self.dimension:
            raise ValueError(f"Query dimension {len(query_embedding)} != {self.dimension}")
        
        if self.index.ntotal == 0:
            return []
        
        vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        vec = self._normalize(vec[0]).reshape(1, -1)
        
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(vec, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for invalid indices
                continue
            if idx not in self.index_to_id:
                continue
            
            similarity = float(dist)  # Already normalized, so this is cosine similarity
            if similarity >= threshold:
                id = self.index_to_id[idx]
                metadata = self.metadata_store.get(id, {})
                results.append((id, similarity, metadata))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def get(self, id: str) -> Optional[Tuple[List[float], dict]]:
        """Get vector and metadata by ID."""
        if id not in self.id_to_index:
            return None
        
        # FAISS doesn't support direct retrieval, so we'd need to store vectors separately
        # For now, return metadata only
        metadata = self.metadata_store.get(id, {})
        return None, metadata  # Vector retrieval not implemented
    
    def delete(self, id: str) -> None:
        """Delete a vector by ID."""
        if id not in self.id_to_index:
            return
        
        idx = self.id_to_index[id]
        self.index.remove_ids(np.array([idx]))
        del self.id_to_index[id]
        del self.index_to_id[idx]
        del self.metadata_store[id]
        logger.debug("Deleted vector", id=id[:8])
    
    def clear(self) -> None:
        """Clear all vectors."""
        self.index = type(self.index)(self.dimension)
        self.id_to_index.clear()
        self.index_to_id.clear()
        self.metadata_store.clear()
        self.next_index = 0
        logger.info("FAISSVectorStore cleared")


class ChromaVectorStore(VectorStore):
    """ChromaDB-based vector store."""
    
    def __init__(self, collection_name: str = "tokenomics_memory", persist_directory: Optional[str] = None):
        """Initialize ChromaDB vector store."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("chromadb is required. Install with: pip install chromadb")
        
        self.client = chromadb.Client(
            Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False,
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except:
            self.collection = self.client.create_collection(name=collection_name)
        
        logger.info("ChromaVectorStore initialized", collection=collection_name)
    
    def add(self, id: str, embedding: List[float], metadata: dict) -> None:
        """Add a vector with metadata."""
        # ChromaDB expects metadata as dict with string values
        chroma_metadata = {k: str(v) for k, v in metadata.items()}
        
        self.collection.add(
            ids=[id],
            embeddings=[embedding],
            metadatas=[chroma_metadata],
        )
        logger.debug("Added vector to ChromaDB", id=id[:8])
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float, dict]]:
        """Search for similar vectors."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        
        output = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i, (id, distance, metadata) in enumerate(zip(
                results["ids"][0],
                results["distances"][0],
                results["metadatas"][0],
            )):
                # ChromaDB returns distance, convert to similarity
                similarity = 1.0 - distance  # Assuming cosine distance
                if similarity >= threshold:
                    # Convert metadata back from strings
                    metadata_dict = {k: v for k, v in metadata.items()}
                    output.append((id, similarity, metadata_dict))
        
        return sorted(output, key=lambda x: x[1], reverse=True)
    
    def get(self, id: str) -> Optional[Tuple[List[float], dict]]:
        """Get vector and metadata by ID."""
        results = self.collection.get(ids=[id])
        if not results["ids"]:
            return None
        
        # ChromaDB doesn't return embeddings in get(), need to query
        # For now, return metadata only
        metadata = {k: v for k, v in results["metadatas"][0].items()}
        return None, metadata
    
    def delete(self, id: str) -> None:
        """Delete a vector by ID."""
        self.collection.delete(ids=[id])
        logger.debug("Deleted vector from ChromaDB", id=id[:8])
    
    def clear(self) -> None:
        """Clear all vectors."""
        # Delete collection and recreate
        try:
            self.client.delete_collection(name=self.collection.name)
        except:
            pass
        self.collection = self.client.create_collection(name=self.collection.name)
        logger.info("ChromaVectorStore cleared")

