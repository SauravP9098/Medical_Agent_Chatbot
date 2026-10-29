from app.knowledge import ensure_index, embed

index = ensure_index()
print("Index connected:", index)

vectors = embed(["this is a test sentence"])
print("Embedding length:", len(vectors[0]))

resp = index.upsert(vectors=[{'id': 'test-1', 'values': vectors[0], 'metadata': {'text': 'hello'}}])
print("Upsert response:", resp)