import argparse
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
import hashlib
from pathlib import Path
import fitz
from app.knowledge import ensure_index, embed

def chunks(text, size=1100, overlap=180):
    text = ' '.join(text.split())
    step = size - overlap
    return [text[i:i+size] for i in range(0, len(text), step) if len(text[i:i+size]) > 80]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', default='data/Medical_book.pdf')
    args = parser.parse_args()
    pdf = Path(args.pdf)
    if not pdf.exists():
        raise SystemExit(f'PDF not found: {pdf}')
    index = ensure_index()
    total_chunks = 0
    with fitz.open(pdf) as doc:
        batch = []
        for page_number, page in enumerate(doc, start=1):
            page_text = page.get_text()
            page_chunks = chunks(page_text)
            if page_number <= 20:
                print(f"[DEBUG] page {page_number}: {len(page_text)} chars extracted, {len(page_chunks)} chunks made", flush=True)
            for part, content in enumerate(page_chunks, start=1):
                key = hashlib.sha256(f'{pdf.name}:{page_number}:{part}:{content}'.encode()).hexdigest()
                batch.append((key, content, {'page': page_number, 'text': content, 'file': pdf.name}))
                total_chunks += 1
                if len(batch) >= 50:
                    vectors = embed([b[1] for b in batch])
                    resp = index.upsert(vectors=[{'id': b[0], 'values': v, 'metadata': b[2]} for b, v in zip(batch, vectors)])
                    print(f"[DEBUG] upserted {resp.upserted_count} vectors, total_chunks so far: {total_chunks}", flush=True)
                    batch = []
            if page_number % 50 == 0:
                print(f'Indexed pages: {page_number}/{len(doc)}', flush=True)
        if batch:
            vectors = embed([b[1] for b in batch])
            resp = index.upsert(vectors=[{'id': b[0], 'values': v, 'metadata': b[2]} for b, v in zip(batch, vectors)])
            print(f"[DEBUG] final upsert {resp.upserted_count} vectors, total_chunks: {total_chunks}", flush=True)
    print(f'Ingestion complete. Total chunks processed: {total_chunks}')

if __name__ == '__main__':
    main()