import statistics

import tiktoken

# scratch_inspect.py (root of repo, don't commit)
from src.ingestion.arxiv_client import ArxivClient

client = ArxivClient()
results = client.get_arxiv_results("machine learning", max_results=3)

# for r in results:
#     text = r.full_text
#     if not text:
#         continue
#     chars = len(text)
#     words = len(text.split())
#     paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 100]

#     print(f"\n{'='*60}")
#     print(f"PAPER: {r.title}")
#     print(f"Total chars: {chars:,} | Words: {words:,} | ~Paragraphs: {len(paragraphs)}")
#     print(f"\n--- START (first 1000 chars) ---")
#     print(text[:1000])
#     print(f"\n--- MIDDLE ---")
#     mid = chars // 2
#     print(text[mid:mid+1000])
#     print(f"\n--- END (last 500 chars) ---")
#     print(text[-500:])


enc = tiktoken.get_encoding("cl100k_base")

for r in results:
    text = r.full_text
    if not text:
        continue

    # Paragraph lengths (the natural semantic unit)
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]

    token_counts = [len(enc.encode(p)) for p in paragraphs]

    print(f"\nPAPER: {r.title[:60]}")
    print(f"  Paragraphs      : {len(paragraphs)}")
    print(f"  Total tokens    : {len(enc.encode(text)):,}")
    print(f"  Min tokens/para : {min(token_counts)}")
    print(f"  Median          : {int(statistics.median(token_counts))}")
    print(f"  Mean            : {int(statistics.mean(token_counts))}")
    print(f"  Max             : {max(token_counts)}")

    # # Print a few sample paragraphs to eyeball coherence
    # print(f"\n  -- Sample paragraphs --")
    # for p in paragraphs[2:5]:   # skip title/authors, grab body
    #     print(f"  [{len(p)} chars] {p[:200]}...\n")
