# Assumptions

- The input PDF is text-based or readable (not heavily corrupted)
- The system runs on a machine with sufficient RAM/GPU for local models
- The invoice image is clear and readable
- English language is used for queries and documents

---

# Limitations

# 1. Context Size Limitation
- The LLM is limited by context window (2048 tokens)
- Long documents may lose information during retrieval

# 2. OCR / Vision Accuracy
- Vision model accuracy depends on image quality
- Complex layouts may reduce extraction accuracy

# 3. Performance Constraints
- Running locally requires high RAM / GPU
- Inference is slower compared to cloud APIs

# 4. No Structured Output Enforcement
- Vision output is plain text (not strict JSON)
- May require post-processing

# 5. Basic Retrieval Strategy
- Uses simple similarity search
- No hybrid (BM25 + semantic) retrieval

---

# Future Improvements

# RAG Enhancements
- Hybrid search (keyword + semantic)
- Reranking (Cross-Encoder)
- Multi-document support
- Metadata filtering

---

# Vision Improvements
- Extract structured JSON fields:
  - invoice_number
  - date
  - total
- Use specialized models (LayoutLM, Donut)

---

# Performance
- Increase context window (n_ctx)
- Use GPU optimization
- Quantized models tuning

---

# UI / UX
- Build a Streamlit interface
- Upload documents dynamically
- Chat interface with history

---

# Integration
- Connect RAG + Vision pipelines
- Store extracted invoice data in vector DB

---

# Final Note

This system is designed as a **fully local, privacy-preserving AI solution**, and serves as a strong foundation for real-world document AI applications.