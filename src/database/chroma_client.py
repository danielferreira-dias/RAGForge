import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
from pathlib import Path


class ChromaDBManager:
    """
    Manages ChromaDB local database for document chunking storage.

    This class provides methods to store, retrieve, and manage document chunks
    with their embeddings in a persistent ChromaDB instance.
    """

    def __init__( self, persist_directory: str = "./chroma_db", collection_name: str = "document_chunks"):
        """
        Initialize ChromaDB client with local persistence.

        Args:
            persist_directory: Path to store the ChromaDB data
            collection_name: Name of the collection to store chunks
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name

        # Create directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Storage for document chunks and embeddings"}
        )

    def add_chunks(self, chunks: List[str], embeddings: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> None:
        """
        Add document chunks to the database.

        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors for each chunk
            metadatas: Optional list of metadata dictionaries for each chunk
            ids: Optional list of unique IDs for each chunk (auto-generated if not provided)
        """
        if ids is None:
            # Generate unique IDs based on collection size
            start_id = self.collection.count()
            ids = [f"chunk_{start_id + i}" for i in range(len(chunks))]

        self.collection.add(ndocuments=chunks,nembeddings=embeddings,nmetadatas=metadatas,nids=ids)

    def query_similar(self,query_embedding: List[float],n_results: int = 10,where: Optional[Dict[str, Any]] = None,where_document: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query for similar chunks based on embedding similarity.

        Args:
            query_embedding: Embedding vector of the query
            n_results: Number of results to return
            where: Optional metadata filter
            where_document: Optional document content filter

        Returns:
            Dictionary containing ids, documents, metadatas, and distances
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document
        )
        return results

    def get_chunks_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """
        Retrieve specific chunks by their IDs.

        Args:
            ids: List of chunk IDs to retrieve

        Returns:
            Dictionary containing the requested chunks with their metadata
        """
        return self.collection.get(ids=ids)

    def delete_chunks(self, ids: List[str]) -> None:
        """
        Delete chunks by their IDs.

        Args:
            ids: List of chunk IDs to delete
        """
        self.collection.delete(ids=ids)

    def count_chunks(self) -> int:
        """
        Get the total number of chunks in the collection.

        Returns:
            Total count of chunks
        """
        return self.collection.count()

    def reset_collection(self) -> None:
        """
        Delete all chunks from the collection.
        Warning: This operation cannot be undone.
        """
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Storage for document chunks and embeddings"}
        )

    def list_collections(self) -> List[str]:
        """
        List all available collections in the database.

        Returns:
            List of collection names
        """
        collections = self.client.list_collections()
        return [col.name for col in collections]

    def switch_collection(self, collection_name: str) -> None:
        """
        Switch to a different collection.

        Args:
            collection_name: Name of the collection to switch to
        """
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Storage for document chunks and embeddings"}
        )

    def update_chunk_metadata( self, ids: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """
        Update metadata for existing chunks.

        Args:
            ids: List of chunk IDs to update
            metadatas: List of new metadata dictionaries
        """
        self.collection.update(
            ids=ids,
            metadatas=metadatas
        )
