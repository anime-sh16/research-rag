import json
from collections import defaultdict
from pathlib import Path

# Use latest file
path = sorted(Path("data/tmp").glob("chunks_*.jsonl"))[-1]
print(f"File: {path.name}")

paper_info = {}  # paper_id -> {title, topic, chunks, authors}

with open(path) as f:
    for line in f:
        c = json.loads(line)
        pid = c["paper_id"]
        if pid not in paper_info:
            paper_info[pid] = {
                "title": c.get("title", ""),
                "topic": c.get("topic", ""),
                "authors": c.get("authors", []),
                "chunks": 0,
            }
        paper_info[pid]["chunks"] += 1

# Group by topic, sort by chunk count descending
by_topic = defaultdict(list)
for pid, info in paper_info.items():
    by_topic[info["topic"]].append((pid, info))

print(f"Total papers: {len(paper_info)}")
print()

for topic in sorted(by_topic.keys()):
    papers = sorted(by_topic[topic], key=lambda x: -x[1]["chunks"])
    total_chunks = sum(p[1]["chunks"] for p in papers)
    print(f"── {topic} ({len(papers)} papers, {total_chunks} chunks) ──")
    for pid, info in papers:
        authors = ", ".join(info["authors"][:2]) + (
            " et al." if len(info["authors"]) > 2 else ""
        )
        flag = " ⚠ abstract-only" if info["chunks"] < 5 else ""
        print(f"  [{info['chunks']:>3}c] {info['title'][:70]:<70} | {authors}{flag}")
    print()
