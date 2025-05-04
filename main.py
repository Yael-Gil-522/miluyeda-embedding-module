import chromadb
from chromadb.utils import embedding_functions

import nltk
nltk.download('punkt')

import fitz  # PyMuPDF
from docx import Document
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# --- קריאת טקסט מ-PDF ---
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"שגיאה בקריאת קובץ PDF: {e}")
    return text

# --- קריאת טקסט מ-DOCX ---
def extract_text_from_docx(file_path):
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"שגיאה בקריאת קובץ DOCX: {e}")
    return text

# --- פיצול לפי משפטים ---
def split_to_sentences(text):
    raw_sentences = text.split('.')
    sentences = [s.strip() + '.' for s in raw_sentences if s.strip()]
    return sentences

# --- פיצול לפי פסקאות ---
def split_to_paragraphs(text):
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return paragraphs

# --- פיצול לפי גודל קבוע עם חפיפה ---
def chunk_fixed_overlap(text, chunk_size=30, overlap=10):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap
    return chunks

# --- הפעלת התהליך ---
if __name__ == "__main__":
    file_path = "sample.docx"  # כאן אפשר לשנות לפי הקובץ שלך

    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        text = extract_text_from_docx(file_path)
    else:
        print("פורמט קובץ לא נתמך.")
        text = ""

    if text:
        print("\n--- טקסט שהופק מהקובץ ---")
        print(text[:500])  # הדפסת חלק מהטקסט בלבד

        # חלוקה
        print("\n--- פיצול למשפטים ---")
        sentences = split_to_sentences(text)
        for i, s in enumerate(sentences):
            print(f"משפט {i+1}: {s}")

        print("\n--- פיצול לפסקאות ---")
        paragraphs = split_to_paragraphs(text)
        for i, p in enumerate(paragraphs):
            print(f"פסקה {i+1}: {p}")

        print("\n--- פיצול לצ'אנקים קבועים ---")
        chunks = chunk_fixed_overlap(text)
        for i, c in enumerate(chunks):
            print(f"Chunk {i+1}: {c}")

        # יצירת Embeddings
        print("\n--- Embeddings למשפטים ---")
        sentence_embeddings = model.encode(sentences)

        for i, (sentence, embedding) in enumerate(zip(sentences, sentence_embeddings)):
            print(f"משפט {i+1}: {sentence}")
            print(f"וקטור (חלקי): {embedding[:5]}...\n")

        # שמירה ב-DB
        chroma_client = chromadb.Client()
        collection = chroma_client.create_collection(name="miluyeda")

        collection.add(
            documents=sentences,
            embeddings=[e.tolist() for e in sentence_embeddings],
            ids=[f"s-{i+1}" for i in range(len(sentences))]
        )

        print("\n--- נשמר בהצלחה ב-ChromaDB ---")

        # חיפוש סמנטי לדוגמה
        query = "מה הקשר בין משפטים סמוכים?"
        query_embedding = model.encode(query)

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=1
        )

        print("\n--- תוצאה מהחיפוש הסמנטי ---")
        print("שאלה:", query)
        print("תוצאה שנשלפה:", results["documents"][0][0])

    else:
        print("לא הצלחנו לחלץ טקסט מהקובץ.")
    