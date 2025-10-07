#!/usr/bin/env python3
"""
Quick script to build RAG database when adding new PDFs
"""

import os
import shutil
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.rag import initialize_rag

def build_rag_database():
    """Clear existing database and build with all PDFs and markdown files"""
    print("Building RAG Database")
    print("=" * 50)
    
    # Clear existing database
    if os.path.exists("chroma_db"):
        print("Clearing existing database...")
        shutil.rmtree("chroma_db")
        print("Database cleared")
    else:
        print("No existing database found")
    
    print("\nBuilding database with all content...")
    try:
        initialize_rag()
        print("Database built successfully!")
        print("\n Content processed:")
        
        # List processed files
        data_dir = "./data/science"
        if os.path.exists(data_dir):
            md_files = [f for f in os.listdir(data_dir) if f.endswith('.md')]
            pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
            
            print(f"  📄 Markdown files: {len(md_files)}")
            for f in md_files:
                print(f"    - {f}")
            
            print(f"  📚 PDF files: {len(pdf_files)}")
            for f in pdf_files:
                print(f"    - {f}")
        
        print(f"\nReady to use! Run 'python test_rag.py demo' to test with chemistry topics")
        
    except Exception as e:
        print(f"Error building database: {e}")
        return False
    
    return True

if __name__ == "__main__":
    build_rag_database()
