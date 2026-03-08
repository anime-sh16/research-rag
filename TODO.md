# TODO

## Next session

- [ ] Expand `IngestionConfig.topics` in `src/config/config.py` to 7–8 topics covering the RAG/ML domain (e.g. transformer architecture, RAG, fine-tuning, RLHF, diffusion models, graph neural networks, etc.)
- [ ] Fetch and download all PDFs for the full topic list — verify PDFs land in `data/pdfs/<topic>/`
- [ ] Update the embedding pipeline (`src/ingestion/vector_store.py`) to handle Gemini API rate limits: exponential backoff + retry logic before embedding each batch
