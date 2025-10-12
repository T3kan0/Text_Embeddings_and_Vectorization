#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 17:03:57 2025

@author: mbonanits
"""

import fitz  # PyMuPDF for PDF processing
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


##### PDF Files for Context #########################################################################

# Files directory (relative or absolute)

folder = Path("/Users/mbonanits/Desktop/A_STEP/2252/Automation/Tutor/Content Docs/Faculty/")

# Get all files (non-recursive)
files = sorted(folder.glob("*"))         # all files
# Or only PDFs:
pdf_files = sorted(folder.glob("*.pdf"))

for p in files:
    print(p)         

#### Function for Extract Text from PDF ##############################################################

def extract_text_from_pdfs(pdf_paths):
    """
    Extracts text content from a list of PDF file paths.

    Args:
        pdf_paths (list[Path]): list of Path objects for PDF files.

    Returns:
        dict: filename (str) -> extracted text (str)
    """
    texts = {}
    for pdf_path in pdf_paths:
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text")
        texts[pdf_path.stem] = text.strip()
    return texts

pdf_texts = extract_text_from_pdfs(pdf_files)

## Extracted Texts from the PDF Files ###################################################################

for name, content in pdf_texts.items():
    print(f"✅ {name}: {len(content)} characters extracted")
    
    
##### Embedding Text into Vectors Function ##############################################################

def create_tfidf_embeddings(pdf_texts):
    """
    Create TF-IDF embeddings for each PDF text.
    Returns a dict: {filename: [{'chunk': chunk_text, 'vector': vector}]}
    """
    all_embeddings = {}
    vectorizer = TfidfVectorizer()

    for filename, text in pdf_texts.items():
        # Optionally, split into chunks if text is very long
        chunks = [text]  # for now, full text as one chunk
        embeddings_list = []

        for chunk in chunks:
            vector = vectorizer.fit_transform([chunk]).toarray()[0]  # Convert sparse to array
            embeddings_list.append({
                "chunk": chunk,
                "vector": vector
            })
        all_embeddings[filename] = embeddings_list

    return all_embeddings


tfidf_embeddings = create_tfidf_embeddings(pdf_texts)

## Extracted Vectors from the PDF Files ###################################################################


rows = []
for filename, emb_list in tfidf_embeddings.items():
    for e in emb_list:
        rows.append({
            "filename": filename,
            "chunk": e["chunk"],
            "embedding": e["vector"].tolist()  # Convert to list for CSV storage
        })

df = pd.DataFrame(rows)
print(df.head())

# Keep only the columns needed for Postgres
df_postgres = df[['filename', 'chunk', 'embedding']]

# Save as new CSV
#df_postgres.to_csv("embeddings_for_postgres.csv", index=False)
#df.to_csv("faculty_tfidf_embeddings.csv", index=False)
#print("✅ Saved TF-IDF embeddings to faculty_tfidf_embeddings.csv")










    