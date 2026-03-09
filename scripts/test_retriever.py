from src.retrieval.retriever import Retriever

r = Retriever(top_k=3)
chunks = r.retrieve("what is retrieval augmented generation")
for c in chunks:
    print(round(c["score"], 3), c["title"])
