# Miluyeda Embedding Module

This module processes PDF and DOCX files, splits their text into semantically meaningful chunks using multiple strategies (fixed-size with overlap, sentence and paragraph-based), generates embeddings using a modern sentence transformer model, and stores them in a local Chroma vector database.

## Features
- Support for PDF and DOCX files
- Sentence, paragraph, and fixed-size overlapping chunking
- Embedding generation with `paraphrase-multilingual-MiniLM-L12-v2`
- Semantic search via ChromaDB

## Files Included
- `main.py` – Main Python module
- `sample.pdf` – Example PDF for testing
- `sample.docx` – Example DOCX for testing
- `README.md` – This description file

## How to Run
1. Install requirements:
   ```bash
   pip install -r requirements.txt
2. Run the module:
   ```bash
   python3 main.py
