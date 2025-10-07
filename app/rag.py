import os
import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pypdf

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import markdown
from sqlalchemy.orm import Session

from .db import Progress, get_current_lesson, create_progress, update_progress

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    """
    RAG (Retrieval-Augmented Generation) service for science education.
    Handles document loading, embedding, and retrieval for age-appropriate science lessons.
    """
    
    def __init__(self, data_dir: str = "./data/science", collection_name: str = "science_lessons"):
        self.data_dir = Path(data_dir)
        self.collection_name = collection_name
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self._initialized = False
        
    def initialize(self):
        """Initialize the RAG service with embedding model and vector database."""
        if self._initialized:
            return
            
        try:
            logger.info("Initializing RAG service...")
            
            # Initialize embedding model
            logger.info("Loading sentence transformer model...")
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Initialize ChromaDB
            logger.info("Initializing ChromaDB...")
            self.chroma_client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
                logger.info(f"Found existing collection: {self.collection_name}")
            except:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Science educational content for ages 6-12"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
            
            # Load and process documents if collection is empty
            if self.collection.count() == 0:
                logger.info("Collection is empty, loading documents...")
                self._load_and_process_documents()
            else:
                logger.info(f"Collection already has {self.collection.count()} documents")
            
            self._initialized = True
            logger.info("RAG service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {str(e)}")
            raise Exception(f"RAG initialization failed: {str(e)}")

    def _load_and_process_documents(self):
        """Load science documents (.md and .pdf) and process them into chunks."""
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist")
            return

        documents, metadatas, ids = [], [], []

        # --- Process Markdown ---
        for file_path in self.data_dir.glob("*.md"):
            self._process_md_file(file_path, documents, metadatas, ids)

        # --- Process PDFs ---
        for file_path in self.data_dir.glob("*.pdf"):
            self._process_pdf_file(file_path, documents, metadatas, ids)

        if documents:
            logger.info(f"Generating embeddings for {len(documents)} chunks...")
            embeddings = self.embedding_model.encode(documents).tolist()
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Successfully added {len(documents)} chunks to collection")
        else:
            logger.warning("No documents were processed")

    def _process_md_file(self, file_path, documents, metadatas, ids):
        """Helper: process Markdown files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            html = markdown.markdown(content)
            text = self._html_to_text(html)
            chunks = self._chunk_text(text)
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 50:
                    documents.append(chunk)
                    metadatas.append({"source": file_path.name, "chunk_id": i,
                                    "topic": self._extract_topic_from_filename(file_path.name)})
                    ids.append(f"{file_path.stem}_{i}")
            logger.info(f"Processed {len(chunks)} chunks from {file_path.name}")
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")

    def _process_pdf_file(self, file_path, documents, metadatas, ids):
        """Helper: process PDF files"""
        try:
            reader = pypdf.PdfReader(file_path)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text() or ""
                full_text += text + "\n"

            chunks = self._chunk_text(full_text)
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 50:
                    documents.append(chunk)
                    metadatas.append({"source": file_path.name, "chunk_id": i,
                                    "topic": self._extract_topic_from_filename(file_path.name)})
                    ids.append(f"{file_path.stem}_{i}")
            logger.info(f"Processed {len(chunks)} chunks from {file_path.name}")
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")

    
    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def _extract_topic_from_filename(self, filename: str) -> str:
        """Extract topic from filename."""
        return filename.replace('.md', '').replace('_', ' ').title()
    
    def retrieve_relevant_chunks(self, query: str, age_group: int, limit: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a given query."""
        if not self._initialized:
            self.initialize()
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search for similar chunks
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=['documents', 'metadatas', 'distances']
            )
            
            chunks = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                chunks.append({
                    'content': doc,
                    'metadata': metadata,
                    'similarity_score': 1 - distance,  # Convert distance to similarity
                    'chunk_id': f"{metadata['source'].replace('.md', '')}_{metadata['chunk_id']}"
                })
            
            logger.info(f"Retrieved {len(chunks)} relevant chunks for query: {query}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            return []
    
    def get_next_chunk(self, topic: str, current_chunk_id: str, age_group: int) -> Optional[Dict[str, Any]]:
        """Get the next chunk for continuing a lesson."""
        if not self._initialized:
            self.initialize()
        
        try:
            # Extract source and chunk number from current_chunk_id
            source, chunk_num = current_chunk_id.rsplit('_', 1)
            current_chunk_num = int(chunk_num)
            
            # Determine the source file name (handle both .md and .pdf)
            if source.endswith('.md') or source.endswith('.pdf'):
                source_file = source
            else:
                # Try to find the actual source file
                source_file = None
                for ext in ['.md', '.pdf']:
                    test_source = f"{source}{ext}"
                    # Check if this source exists in the collection
                    test_results = self.collection.query(
                        query_texts=[topic],
                        n_results=1,
                        where={"source": test_source},
                        include=['documents', 'metadatas']
                    )
                    if test_results['documents'][0]:
                        source_file = test_source
                        break
            
            if not source_file:
                logger.error(f"Could not find source file for {source}")
                return None
            
            # Search for chunks from the same source
            results = self.collection.query(
                query_texts=[topic],
                n_results=100,  # Get more results to find the next chunk
                where={"source": source_file},
                include=['documents', 'metadatas']
            )
            
            # Find the next chunk by looking for the specific chunk_id
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                if metadata['chunk_id'] == current_chunk_num + 1:
                    return {
                        'content': doc,
                        'metadata': metadata,
                        'chunk_id': f"{source}_{metadata['chunk_id']}"
                    }
            
            # If no immediate next chunk found, try to get any remaining chunks from the same source
            # This handles cases where chunks might not be in sequential order
            available_chunks = []
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                if metadata['chunk_id'] > current_chunk_num:
                    available_chunks.append((metadata['chunk_id'], doc, metadata))
            
            if available_chunks:
                # Sort by chunk_id and return the next one
                available_chunks.sort(key=lambda x: x[0])
                next_chunk_id, doc, metadata = available_chunks[0]
                return {
                    'content': doc,
                    'metadata': metadata,
                    'chunk_id': f"{source}_{next_chunk_id}"
                }
            
            # If no next chunk found, return None
            return None
            
        except Exception as e:
            logger.error(f"Error getting next chunk: {str(e)}")
            return None
    
    def create_rag_lesson_prompt(self, topic: str, retrieved_chunks: List[Dict[str, Any]], 
                                age_group: int, user_name: str = "") -> Tuple[str, str]:
        """Create a prompt for generating a RAG-based lesson."""
        
        # Age-appropriate style guide
        if age_group <= 8:
            style_guide = "Use very simple words, short sentences, and examples with toys, animals, or games. Use lots of emojis and make it fun!"
        elif age_group <= 10:
            style_guide = "Use simple language, clear examples, and everyday situations like school or home. Include some emojis and make it engaging!"
        else:
            style_guide = "Use clear explanations with relatable examples and real-world situations. Make it interesting and educational!"
        
        # Prepare context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(f"Source {i}:\n{chunk['content']}\n")
        
        context = "\n".join(context_parts)
        
        system_prompt = f"""You are an expert science teacher for children aged {age_group} years old.
Your goal is to create an engaging, accurate science lesson using the provided educational content.

Instructions:
- Topic: {topic}
- Age group: {age_group} years old
- Length: Keep it concise (150-250 words max)
- Style: {style_guide}
- Use the provided educational content as your source of information
- Make sure all facts are accurate and age-appropriate
- Structure: Brief introduction, key explanation, fun example, and one simple question

Important: Base your lesson on the provided educational content. Do not make up facts that aren't in the source material."""

        greeting = f"Hey {user_name}! " if user_name else ""
        user_prompt = f"""{greeting}Please teach me about {topic} using this educational content:

{context}

Create a fun, engaging lesson that helps me understand {topic} better!"""

        return system_prompt, user_prompt

# Global RAG service instance
rag_service = RAGService()

def initialize_rag():
    """Initialize the RAG service."""
    rag_service.initialize()

def get_rag_lesson(topic: str, age_group: int, user_name: str = "", 
                  current_chunk_id: str = None) -> Tuple[str, str]:
    """
    Generate a RAG-based lesson for the given topic.
    Returns (lesson_content, chunk_id) for progress tracking.
    """
    if not rag_service._initialized:
        rag_service.initialize()
    
    # If continuing a lesson, try to get next chunk
    if current_chunk_id:
        next_chunk = rag_service.get_next_chunk(topic, current_chunk_id, age_group)
        if next_chunk:
            retrieved_chunks = [next_chunk]
            chunk_id = next_chunk['chunk_id']
        else:
            # No more chunks, retrieve fresh content
            retrieved_chunks = rag_service.retrieve_relevant_chunks(topic, age_group)
            chunk_id = retrieved_chunks[0]['chunk_id'] if retrieved_chunks else None
    else:
        # New lesson, retrieve relevant chunks
        retrieved_chunks = rag_service.retrieve_relevant_chunks(topic, age_group)
        chunk_id = retrieved_chunks[0]['chunk_id'] if retrieved_chunks else None
    
    if not retrieved_chunks:
        return "I'm sorry, I couldn't find information about that topic in my science database. Try asking about plants, animals, the solar system, energy, or weather!", None
    
    # Create prompt for LLM
    system_prompt, user_prompt = rag_service.create_rag_lesson_prompt(
        topic, retrieved_chunks, age_group, user_name
    )
    
    return system_prompt, user_prompt, chunk_id

if __name__ == "__main__":
    # Test the RAG service
    print("Testing RAG service...")
    initialize_rag()
    
    # Test retrieval
    chunks = rag_service.retrieve_relevant_chunks("plants and photosynthesis", 8)
    print(f"Retrieved {len(chunks)} chunks")
    for chunk in chunks:
        print(f"Chunk: {chunk['chunk_id']}")
        print(f"Content: {chunk['content'][:100]}...")
        print(f"Similarity: {chunk['similarity_score']:.3f}")
        print("-" * 50)
