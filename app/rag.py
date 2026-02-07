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
from sentence_transformers import CrossEncoder

from .db import Progress, get_current_lesson, create_progress, update_progress

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, chroma_client, collection_name, embedding_model_name='sentence-transformers/all-MiniLM-L6-v2', use_reranker=True):
        self.chroma_client = chroma_client
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.use_reranker = use_reranker

        # if self.use_reranker:
        #     self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

class RAGService:
    """
    RAG (Retrieval-Augmented Generation) service for science education.
    Handles document loading, embedding, and retrieval for age-appropriate science lessons.
    """
    
    def __init__(self, data_dir: str = "./data/science", collection_name: str = "science_lessons", use_reranker: bool = True):
        self.data_dir = Path(data_dir)
        self.collection_name = collection_name
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self._initialized = False
        self.use_reranker = use_reranker
        self.reranker = None
        
    def initialize(self):
        """Initialize the RAG service with embedding model and vector database."""
        if self._initialized:
            return
            
        try:
            logger.info("Initializing RAG service...")
            
            logger.info("Loading sentence transformer model...")
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            if self.use_reranker:
                logger.info("Loading cross-encoder reranker...")
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            logger.info("Initializing ChromaDB...")
            self.chroma_client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(anonymized_telemetry=False)
            )
            
            try:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
                logger.info(f"Found existing collection: {self.collection_name}")
            except:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Science educational content for ages 6-12"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
            
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
    
    def _chunk_text(self, text: str, chunk_size: int = 200, overlap: int = 60) -> List[str]:
        """Split text into overlapping chunks.
        """
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
    
    def retrieve_relevant_chunks(self, query: str, limit: int = 5):
        # Step 1: Embed the query
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)

        # Step 2: Query Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit * 3, 
            include=['documents', 'metadatas', 'distances']
        )

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        # Step 3: Normalize cosine distance → similarity score
        similarities = [(1 - d / 2) for d in distances]  # for cosine distance in [0 to 2]
        results_with_scores = list(zip(documents, metadatas, similarities))

        # Step 4: Optional re-ranking with cross-encoder
        if self.use_reranker:
            # Prepare pairs (query, doc)
            pairs = [(query, doc) for doc, _, _ in results_with_scores]
            rerank_scores = self.reranker.predict(pairs)
            
            # Normalize rerank scores to [0, 1] range
            min_rerank = min(rerank_scores)
            max_rerank = max(rerank_scores)
            if max_rerank > min_rerank:
                normalized_rerank_scores = [(score - min_rerank) / (max_rerank - min_rerank) for score in rerank_scores]
            else:
                normalized_rerank_scores = [0.5] * len(rerank_scores)  # All same score
            
            # Combine both scores (weighted)
            rerank_weight = 0.7  # 70% re-ranker, 30% embedding similarity
            for i, (doc, meta, sim) in enumerate(results_with_scores):
                combined_score = rerank_weight * normalized_rerank_scores[i] + (1 - rerank_weight) * sim
                results_with_scores[i] = (doc, meta, combined_score)

        # Step 5: Sort by combined score (descending)
        ranked_results = sorted(results_with_scores, key=lambda x: x[2], reverse=True)

        # Step 6: Convert to dictionary and return top-k final results
        chunks = []
        for doc, metadata, similarity_score in ranked_results[:limit]:
            chunks.append({
                'content': doc,
                'metadata': metadata,
                'similarity_score': similarity_score,
                'chunk_id': f"{metadata['source'].replace('.md', '').replace('.pdf', '')}_{metadata['chunk_id']}"
            })
        
        return chunks
    
    def get_next_chunk(self, topic: str, current_chunk_id: str, age_group: int) -> Optional[Dict[str, Any]]:
        """Get the next chunk for continuing a lesson."""
        if not self._initialized:
            self.initialize()
        
        try:
            # Extract source and chunk number from current_chunk_id
            source, chunk_num = current_chunk_id.rsplit('_', 1)
            current_chunk_num = int(chunk_num)
            
            if source.endswith('.md') or source.endswith('.pdf'):
                source_file = source
            else:
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
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting next chunk: {str(e)}")
            return None
    
    def create_rag_lesson_prompt(self, topic: str, retrieved_chunks: List[Dict[str, Any]], 
                                age_group: int, user_name: str = "", 
                                is_continuation: bool = False, previous_content: str = None,
                                for_audio: bool = False) -> Tuple[str, str]:
        """Create a prompt for generating a RAG-based lesson. When for_audio is True, instructs LLM to write for TTS."""
        
        if age_group <= 8:
            style_guide = "Use very simple words, short sentences, and examples with toys, animals, or games. Use lots of emojis and make it fun!"
        elif age_group <= 10:
            style_guide = "Use simple language, clear examples, and everyday situations like school or home. Include some emojis and make it engaging!"
        else:
            style_guide = "Use clear explanations with relatable examples and real-world situations. Make it interesting and educational!"
        
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(f"Source {i}:\n{chunk['content']}\n")
        
        context = "\n".join(context_parts)
        
        if is_continuation and previous_content:
            # Continuation lesson - reference previous content, no example at start
            system_prompt = f"""You are an expert science teacher for children aged {age_group} years old.
You are continuing a lesson on {topic}. The student has already learned the previous part.

Instructions:
- Topic: {topic} (continuation)
- Age group: {age_group} years old
- Length: Keep it concise (under 1400 characters total)
- Style: {style_guide}
- Use the provided educational content as your source of information
- Make sure all facts are accurate and age-appropriate

CONTINUATION STRUCTURE:
- Start by briefly referencing what was covered in the previous part (1-2 sentences)
- Then continue with new information from the provided content
- Do NOT repeat examples from the previous part
- Do NOT start with a new example - jump straight into continuing the explanation
- Make it feel like a natural conversation continuation

CRITICAL FORMATTING RULES:
- Use single asterisk *text* for bold (WhatsApp format), NOT double asterisks **
- Do NOT include "Try This at Home" or similar activity sections unless they directly relate to the topic
- Focus on clear explanations and examples, not generic activities
- Do not add unnecessary formatting or redundant bold markers

CRITICAL COMPLETENESS REQUIREMENTS:
- ALWAYS complete your response with proper ending punctuation (. ! ?)
- NEVER cut off mid-sentence, mid-list, or mid-thought
- If listing items, complete the entire list before ending
- Ensure the response is a complete, coherent continuation that can stand alone
- End naturally with a complete sentence

Important: 
- Base your lesson on the provided educational content. Do not make up facts that aren't in the source material.
- Keep the response under 1400 characters to ensure WhatsApp delivery.
- Make it conversational and connected to what the student just learned.
- ALWAYS provide a complete, finished response."""
            if for_audio:
                system_prompt += """

AUDIO/TTS MODE: Your reply will be read aloud by text-to-speech. Write for listening: use short, complete sentences; avoid markdown headers (##) and bullet lists—use flowing prose instead; do not include instructions like "Type /next"; use minimal or no emojis; write as if you are speaking to the student."""

            user_prompt = f"""Continue teaching about {topic}. 

Previous part of the lesson:
{previous_content[:500]}

New educational content to use:
{context}

Continue the lesson naturally, referencing what was just covered and building on it!"""
        else:
            # New lesson - standard structure
            system_prompt = f"""You are an expert science teacher for children aged {age_group} years old.
Your goal is to create an engaging, accurate science lesson using the provided educational content.

Instructions:
- Topic: {topic}
- Age group: {age_group} years old
- Length: Keep it concise (under 1400 characters total)
- Style: {style_guide}
- Use the provided educational content as your source of information
- Make sure all facts are accurate and age-appropriate
- Structure: Brief introduction, key explanation and a fun example

CRITICAL FORMATTING RULES:
- Use single asterisk *text* for bold (WhatsApp format), NOT double asterisks **
- Do NOT include "Try This at Home" or similar activity sections unless they directly relate to the topic
- Focus on clear explanations and examples, not generic activities
- Do not add unnecessary formatting or redundant bold markers

CRITICAL COMPLETENESS REQUIREMENTS:
- ALWAYS complete your response with proper ending punctuation (. ! ?) BEFORE any emojis
- If you use emojis at the end, place them AFTER the final punctuation mark
- NEVER cut off mid-sentence, mid-list, or mid-thought
- If listing items, complete the entire list before ending
- Ensure the response is a complete, coherent lesson that can stand alone
- End naturally with a complete sentence followed by punctuation, then optional emojis

Important: 
- Base your lesson on the provided educational content. Do not make up facts that aren't in the source material.
- Keep the response under 1400 characters to ensure WhatsApp delivery.
- ALWAYS provide a complete, finished response."""
            if for_audio:
                system_prompt += """

AUDIO/TTS MODE: Your reply will be read aloud by text-to-speech. Write for listening: use short, complete sentences; avoid markdown headers (##) and bullet lists—use flowing prose instead; do not include instructions like "Type /next"; use minimal or no emojis; write as if you are speaking to the student."""

            user_prompt = f"""Please teach me about {topic} using this educational content:

{context}

Create a fun, engaging lesson that helps me understand {topic} better!"""

        return system_prompt, user_prompt

rag_service = RAGService()

def initialize_rag():
    """Initialize the RAG service."""
    rag_service.initialize()

def get_rag_lesson(topic: str, age_group: int, user_name: str = "", 
                  current_chunk_id: str = None, previous_content: str = None,
                  for_audio: bool = False) -> Tuple[str, str, Optional[str]]:
    """
    Generate a RAG-based lesson for the given topic.
    Returns (system_prompt, user_prompt, chunk_id) for lesson generation.
    
    Args:
        topic: The lesson topic
        age_group: Age of the student
        user_name: Name of the student
        current_chunk_id: If continuing a lesson, the current chunk ID
        previous_content: If continuing a lesson, the previous lesson content
        for_audio: If True, prompt asks for spoken-style output (TTS-friendly)
    """
    if not rag_service._initialized:
        rag_service.initialize()
    
    is_continuation = current_chunk_id is not None
    
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
        return "I'm sorry, I couldn't find information about that topic in my science database. Try asking about plants, animals, the solar system, energy, or weather!", None, None
    
    system_prompt, user_prompt = rag_service.create_rag_lesson_prompt(
        topic, retrieved_chunks, age_group, user_name, 
        is_continuation=is_continuation, previous_content=previous_content,
        for_audio=for_audio
    )
    
    return system_prompt, user_prompt, chunk_id

if __name__ == "__main__":
    print("Testing RAG service...")
    initialize_rag()
    
    chunks = rag_service.retrieve_relevant_chunks("plants and photosynthesis", 8)
    print(f"Retrieved {len(chunks)} chunks")
    for chunk in chunks:
        print(f"Chunk: {chunk['chunk_id']}")
        print(f"Content: {chunk['content'][:100]}...")
        print(f"Similarity: {chunk['similarity_score']:.3f}")
        print("-" * 50)
