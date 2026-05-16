from app.ingest import get_vector_store

vs = get_vector_store()
results = vs.similarity_search("attention mechanism transformer architecture", k=3)

for i, r in enumerate(results):
    print(f"Result {i+1} - Page {r.metadata.get('page', '?')}")
    print(r.page_content[:200])
    print()