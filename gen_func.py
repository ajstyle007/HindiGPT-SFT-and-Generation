import re
import fitz
import numpy as np
import torch
import faiss
import sentencepiece as spm
from sentence_transformers import SentenceTransformer, CrossEncoder

sp = spm.SentencePieceProcessor()
sp.load("hindi_tokenizer_new.model")

# 2. Load embedding & reranker
embed_model = SentenceTransformer("intfloat/multilingual-e5-base")
reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

def hard_clean(text):
    text = re.sub(r"[^\u0900-\u097F0-9A-Za-z\s.,!?।:-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_hindi_text(text):
    if not text:
        return ""
    
    # Remove non-printable characters
    text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)

    # Fix common PDF junk chars
    text = re.sub(r'[�•ﬁﬂ–—]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def fix_pdf_text(raw_text):
    text = re.sub(r'\s+', ' ', raw_text)
    text = re.sub(r'([ऀ-ॿ])([A-Za-z0-9])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z0-9])([ऀ-ॿ])', r'\1 \2', text)
    return text.strip()

def chunk_text(text, max_tokens=200, overlap=50):
    tokens = sp.encode(text)
    chunks = []

    i=0
    while i < len(tokens):
        chunk = tokens[i: i+max_tokens]
        chunks.append(sp.decode(chunk))
        i += max_tokens - overlap
    return chunks


def sentence_chunk_text(text, max_chars=800, overlap_sentences=1):
    sentences = re.split(r'(?<=[।!?])\s+', text)
    chunks = []
    current_sents = []

    for sent in sentences:
        current_sents.append(sent)
        chunk_text = " ".join(current_sents)

        if len(chunk_text) >= max_chars:
            chunks.append(chunk_text.strip())
            current_sents = current_sents[-overlap_sentences:]
    
    if current_sents:
        chunks.append(" ".join(current_sents).strip())
    
    return chunks

def build_index(chunks):
    texts = ["passage: " + chunk for chunk in chunks]
    embeddings = embed_model.encode(texts, normalize_embeddings=True, batch_size=32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index, chunks

