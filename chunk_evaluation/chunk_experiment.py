#!/usr/bin/env python3
"""
Chunk Size and Overlap Experiment for RAG System
Tests different chunk sizes (150, 200, 400, 500 tokens) and overlap percentages (10%, 20%, 30%)
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import pypdf
import markdown
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleRAGService:
    """Simplified RAG service for experimentation"""
    
    def __init__(self, data_dir: str = "./data/science", collection_name: str = "science_lessons", 
                 chunk_size: int = 400, overlap_percentage: int = 20):
        self.data_dir = Path(data_dir)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.overlap_percentage = overlap_percentage
        self.overlap_tokens = int(chunk_size * overlap_percentage / 100)
        
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self._initialized = False
        
    def initialize(self):
        """Initialize the RAG service"""
        if self._initialized:
            return
            
        try:
            logger.info(f"Initializing RAG service with chunk_size={self.chunk_size}, overlap={self.overlap_percentage}%")
            
            # Load embedding model
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Initialize ChromaDB
            self.chroma_client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create or get collection
            try:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
                logger.info(f"Found existing collection: {self.collection_name}")
            except:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": f"Science content with chunk_size={self.chunk_size}, overlap={self.overlap_percentage}%"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
            
            # Load documents if collection is empty
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
        """Load and process documents"""
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist")
            return

        documents, metadatas, ids = [], [], []

        # Process Markdown files
        for file_path in self.data_dir.glob("*.md"):
            self._process_md_file(file_path, documents, metadatas, ids)

        # Process PDF files
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
        """Process Markdown files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            html = markdown.markdown(content)
            text = self._html_to_text(html)
            chunks = self._chunk_text(text)
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 50:
                    documents.append(chunk)
                    metadatas.append({
                        "source": file_path.name, 
                        "chunk_id": i,
                        "topic": self._extract_topic_from_filename(file_path.name),
                        "chunk_size": self.chunk_size,
                        "overlap_percentage": self.overlap_percentage
                    })
                    ids.append(f"{file_path.stem}_{i}")
            
            logger.info(f"Processed {len(chunks)} chunks from {file_path.name}")
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")
    
    def _process_pdf_file(self, file_path, documents, metadatas, ids):
        """Process PDF files"""
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
                    metadatas.append({
                        "source": file_path.name, 
                        "chunk_id": i,
                        "topic": self._extract_topic_from_filename(file_path.name),
                        "chunk_size": self.chunk_size,
                        "overlap_percentage": self.overlap_percentage
                    })
                    ids.append(f"{file_path.stem}_{i}")
            
            logger.info(f"Processed {len(chunks)} chunks from {file_path.name}")
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")
    
    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text"""
        text = re.sub(r'<[^>]+>', '', html)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap_tokens):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def _extract_topic_from_filename(self, filename: str) -> str:
        """Extract topic from filename"""
        return filename.replace('.md', '').replace('.pdf', '').replace('_', ' ').title()
    
    def retrieve_relevant_chunks(self, query: str, limit: int = 5):
        """Retrieve relevant chunks for a query"""
        if not self._initialized:
            self.initialize()
        
        # Embed the query
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)

        # Query Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit * 3, 
            include=['documents', 'metadatas', 'distances']
        )

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        # Convert cosine distance to similarity score
        similarities = [(1 - d / 2) for d in distances]
        results_with_scores = list(zip(documents, metadatas, similarities))

        # Sort by similarity score
        ranked_results = sorted(results_with_scores, key=lambda x: x[2], reverse=True)

        # Convert to dictionary format
        chunks = []
        for doc, metadata, similarity_score in ranked_results[:limit]:
            chunks.append({
                'content': doc,
                'metadata': metadata,
                'similarity_score': similarity_score,
                'chunk_id': f"{metadata['source'].replace('.md', '').replace('.pdf', '')}_{metadata['chunk_id']}"
            })
        
        return chunks

class ChunkExperiment:
    """Experiment class to test different chunk sizes and overlaps"""
    
    def __init__(self):
        self.results = []
        self.test_queries = [
            "How do plants transport water from roots to leaves?",
            "What is photosynthesis and how does it work?",
            "Explain the process of transpiration in plants",
            "What are the different parts of a plant cell?",
            "How do plants make their own food?",
            "What is the role of chlorophyll in plants?",
            "How do plants absorb water and nutrients?",
            "What happens during plant respiration?",
            "Explain the water cycle in plants",
            "What are stomata and what do they do?"
        ]
        
        # Test configurations
        self.chunk_sizes = [150, 200, 400, 500]  # tokens
        self.overlap_percentages = [10, 20, 30]  # percentage
        
    def evaluate_retrieval_quality(self, rag_service, query: str) -> Dict[str, Any]:
        """Evaluate the quality of retrieval for a given query"""
        
        try:
            # Retrieve chunks
            chunks = rag_service.retrieve_relevant_chunks(query, limit=5)
            
            if not chunks:
                return {
                    'query': query,
                    'num_chunks': 0,
                    'avg_similarity': 0.0,
                    'max_similarity': 0.0,
                    'min_similarity': 0.0,
                    'coverage_score': 0.0,
                    'relevance_score': 0.0
                }
            
            # Extract similarity scores
            similarities = [chunk['similarity_score'] for chunk in chunks]
            
            # Calculate metrics
            avg_similarity = np.mean(similarities)
            max_similarity = np.max(similarities)
            min_similarity = np.min(similarities)
            
            # Coverage score: how many chunks have good similarity (>0.7)
            coverage_score = sum(1 for s in similarities if s > 0.7) / len(similarities)
            
            # Relevance score: weighted average with higher weight for top results
            weights = [1.0, 0.8, 0.6, 0.4, 0.2][:len(similarities)]
            relevance_score = np.average(similarities, weights=weights)
            
            return {
                'query': query,
                'num_chunks': len(chunks),
                'avg_similarity': avg_similarity,
                'max_similarity': max_similarity,
                'min_similarity': min_similarity,
                'coverage_score': coverage_score,
                'relevance_score': relevance_score,
                'similarities': similarities
            }
            
        except Exception as e:
            logger.error(f"Error evaluating query '{query}': {str(e)}")
            return {
                'query': query,
                'num_chunks': 0,
                'avg_similarity': 0.0,
                'max_similarity': 0.0,
                'min_similarity': 0.0,
                'coverage_score': 0.0,
                'relevance_score': 0.0,
                'error': str(e)
            }
    
    def run_experiment(self):
        """Run the complete experiment"""
        
        logger.info("Starting chunk size and overlap experiment...")
        
        for chunk_size in self.chunk_sizes:
            for overlap_percentage in self.overlap_percentages:
                logger.info(f"Testing chunk_size={chunk_size}, overlap={overlap_percentage}%")
                
                try:
                    # Create RAG service with specific configuration
                    collection_name = f"experiment_{chunk_size}_{overlap_percentage}"
                    rag_service = SimpleRAGService(
                        collection_name=collection_name,
                        chunk_size=chunk_size,
                        overlap_percentage=overlap_percentage
                    )
                    
                    # Initialize the service
                    start_time = time.time()
                    rag_service.initialize()
                    init_time = time.time() - start_time
                    
                    logger.info(f"Initialized RAG service in {init_time:.2f} seconds")
                    
                    # Test each query
                    query_results = []
                    for query in self.test_queries:
                        result = self.evaluate_retrieval_quality(rag_service, query)
                        query_results.append(result)
                        time.sleep(0.1)  # Small delay
                    
                    # Calculate overall metrics
                    all_similarities = []
                    all_coverage_scores = []
                    all_relevance_scores = []
                    
                    for result in query_results:
                        if 'similarities' in result:
                            all_similarities.extend(result['similarities'])
                        all_coverage_scores.append(result['coverage_score'])
                        all_relevance_scores.append(result['relevance_score'])
                    
                    # Store results
                    experiment_result = {
                        'chunk_size': chunk_size,
                        'overlap_percentage': overlap_percentage,
                        'overlap_tokens': int(chunk_size * overlap_percentage / 100),
                        'init_time': init_time,
                        'total_chunks': rag_service.collection.count(),
                        'avg_similarity': np.mean(all_similarities) if all_similarities else 0.0,
                        'max_similarity': np.max(all_similarities) if all_similarities else 0.0,
                        'min_similarity': np.min(all_similarities) if all_similarities else 0.0,
                        'avg_coverage_score': np.mean(all_coverage_scores),
                        'avg_relevance_score': np.mean(all_relevance_scores),
                        'query_results': query_results
                    }
                    
                    self.results.append(experiment_result)
                    
                    logger.info(f"Completed chunk_size={chunk_size}, overlap={overlap_percentage}%")
                    logger.info(f"  - Total chunks: {experiment_result['total_chunks']}")
                    logger.info(f"  - Avg similarity: {experiment_result['avg_similarity']:.3f}")
                    logger.info(f"  - Avg coverage: {experiment_result['avg_coverage_score']:.3f}")
                    logger.info(f"  - Avg relevance: {experiment_result['avg_relevance_score']:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error in experiment chunk_size={chunk_size}, overlap={overlap_percentage}%: {str(e)}")
                    continue
        
        logger.info("Experiment completed!")
        return self.results
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze the experiment results and find the best configuration"""
        
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Create DataFrame for easier analysis
        df_data = []
        for result in self.results:
            df_data.append({
                'chunk_size': result['chunk_size'],
                'overlap_percentage': result['overlap_percentage'],
                'overlap_tokens': result['overlap_tokens'],
                'total_chunks': result['total_chunks'],
                'avg_similarity': result['avg_similarity'],
                'avg_coverage_score': result['avg_coverage_score'],
                'avg_relevance_score': result['avg_relevance_score'],
                'init_time': result['init_time']
            })
        
        df = pd.DataFrame(df_data)
        
        # Calculate composite score
        df['composite_score'] = (
            0.4 * df['avg_similarity'] +
            0.3 * df['avg_coverage_score'] +
            0.3 * df['avg_relevance_score']
        )
        
        # Find best configuration
        best_config = df.loc[df['composite_score'].idxmax()]
        
        analysis = {
            'best_configuration': {
                'chunk_size': int(best_config['chunk_size']),
                'overlap_percentage': int(best_config['overlap_percentage']),
                'overlap_tokens': int(best_config['overlap_tokens']),
                'composite_score': float(best_config['composite_score']),
                'avg_similarity': float(best_config['avg_similarity']),
                'avg_coverage_score': float(best_config['avg_coverage_score']),
                'avg_relevance_score': float(best_config['avg_relevance_score']),
                'total_chunks': int(best_config['total_chunks']),
                'init_time': float(best_config['init_time'])
            },
            'all_results': df.to_dict('records'),
            'summary_stats': {
                'total_experiments': len(df),
                'avg_composite_score': float(df['composite_score'].mean()),
                'std_composite_score': float(df['composite_score'].std()),
                'best_similarity': float(df['avg_similarity'].max()),
                'best_coverage': float(df['avg_coverage_score'].max()),
                'best_relevance': float(df['avg_relevance_score'].max())
            }
        }
        
        return analysis
    
    def save_results(self, filename: str = None):
        """Save experiment results to file"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chunk_experiment_results_{timestamp}.json"
        
        save_data = {
            'experiment_config': {
                'chunk_sizes': self.chunk_sizes,
                'overlap_percentages': self.overlap_percentages,
                'test_queries': self.test_queries,
                'timestamp': datetime.now().isoformat()
            },
            'results': self.results,
            'analysis': self.analyze_results()
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filename}")
        return filename

def main():
    """Main function to run the experiment"""
    
    print("🧪 Chunk Size and Overlap Experiment for RAG System")
    print("=" * 60)
    
    # Create experiment instance
    experiment = ChunkExperiment()
    
    # Run the experiment
    print("Starting experiment...")
    results = experiment.run_experiment()
    
    # Analyze results
    print("\nAnalyzing results...")
    analysis = experiment.analyze_results()
    
    # Display results
    print("\n📊 EXPERIMENT RESULTS")
    print("=" * 60)
    
    if 'best_configuration' in analysis:
        best = analysis['best_configuration']
        print(f"🏆 BEST CONFIGURATION:")
        print(f"   Chunk Size: {best['chunk_size']} tokens")
        print(f"   Overlap: {best['overlap_percentage']}% ({best['overlap_tokens']} tokens)")
        print(f"   Composite Score: {best['composite_score']:.3f}")
        print(f"   Avg Similarity: {best['avg_similarity']:.3f}")
        print(f"   Avg Coverage: {best['avg_coverage_score']:.3f}")
        print(f"   Avg Relevance: {best['avg_relevance_score']:.3f}")
        print(f"   Total Chunks: {best['total_chunks']}")
        print(f"   Init Time: {best['init_time']:.2f}s")
    
    print(f"\n📈 SUMMARY STATISTICS:")
    summary = analysis['summary_stats']
    print(f"   Total Experiments: {summary['total_experiments']}")
    print(f"   Avg Composite Score: {summary['avg_composite_score']:.3f}")
    print(f"   Best Similarity: {summary['best_similarity']:.3f}")
    print(f"   Best Coverage: {summary['best_coverage']:.3f}")
    print(f"   Best Relevance: {summary['best_relevance']:.3f}")
    
    # Show all results in a table
    print(f"\n📋 ALL RESULTS:")
    print("-" * 80)
    print(f"{'Chunk':<6} {'Overlap':<8} {'Tokens':<7} {'Chunks':<7} {'Similarity':<11} {'Coverage':<9} {'Relevance':<10} {'Score':<7}")
    print("-" * 80)
    
    for result in analysis['all_results']:
        print(f"{result['chunk_size']:<6} {result['overlap_percentage']:<8}% "
              f"{result['overlap_tokens']:<7} {result['total_chunks']:<7} "
              f"{result['avg_similarity']:<11.3f} {result['avg_coverage_score']:<9.3f} "
              f"{result['avg_relevance_score']:<10.3f} {result['composite_score']:<7.3f}")
    
    # Save results
    filename = experiment.save_results()
    print(f"\n💾 Results saved to: {filename}")
    
    return analysis

if __name__ == "__main__":
    main()
