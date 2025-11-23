import os
import re
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PDF_DIR = "pdfs"

STOP_PHRASES = [
    "p.t.o", "pto", "please turn over"
]

def is_garbage_line(line: str) -> bool:
    t = line.strip().lower()
    if not t:
        return True
    for phrase in STOP_PHRASES:
        if phrase in t:
            return True
    return False

# treat these as "start of a question / subquestion"
QUESTION_START_RE = re.compile(
    r"""^(
        q[\.\s]*\d+      |   # Q3, Q.3, Q 3
        question\s*\d+   |   # Question 3
        \d+\.\s*         |   # 3.
        \d+\)\s*         |   # 3)
        [a-h]\)\s*           # a) b) c) ...
    )""",
    re.IGNORECASE | re.VERBOSE
)

def is_question_start(line: str) -> bool:
    return bool(QUESTION_START_RE.match(line.strip()))

def load_pdfs_to_chunks(pdf_dir):
    chunks = []
    sources = []
    for fname in os.listdir(pdf_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_dir, fname)
        with pdfplumber.open(path) as pdf:
            for page_index, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                raw_lines = text.split("\n")

                lines = []
                for l in raw_lines:
                    l = l.strip()
                    if is_garbage_line(l):
                        continue
                    lines.append(l)

                # group into question blocks
                i = 0
                n = len(lines)
                while i < n:
                    line = lines[i]

                    # start new block at question-like line or at very beginning
                    if is_question_start(line) or i == 0:
                        block_lines = [line]
                        j = i + 1
                        # keep adding lines until next major question start
                        while j < n and not is_question_start(lines[j]):
                            block_lines.append(lines[j])
                            j += 1
                        block = " ".join(block_lines)
                        if len(block) >= 25:
                            chunks.append(block)
                            sources.append(f"{fname} - page {page_index + 1}")
                        i = j
                    else:
                        i += 1
    return chunks, sources

chunks, sources = load_pdfs_to_chunks(PDF_DIR)
if not chunks:
    print("No text found in PDFs in 'pdfs' folder.")
    raise SystemExit

lower_chunks = [c.lower() for c in chunks]

vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(chunks)

def search(query, top_k=5):
    q_lower = query.lower().strip()
    words = [w for w in q_lower.split() if len(w) > 2]

    phrase_idxs = [i for i, t in enumerate(lower_chunks) if q_lower in t]

    if phrase_idxs:
        candidate_idxs = phrase_idxs
    elif words:
        candidate_idxs = [
            i for i, t in enumerate(lower_chunks)
            if all(w in t for w in words)
        ]
    else:
        candidate_idxs = list(range(len(chunks)))

    if not candidate_idxs:
        candidate_idxs = list(range(len(chunks)))

    sub_vectors = doc_vectors[candidate_idxs]
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, sub_vectors)[0]

    ranked_local = sims.argsort()[::-1][:top_k]
    results = []
    for local_idx in ranked_local:
        global_idx = candidate_idxs[local_idx]
        results.append((chunks[global_idx], sources[global_idx], sims[local_idx]))
    return results

print("PYQ chatbot ready. Type 'exit' to quit.")
while True:
    q = input("\nEnter topic name from syllabus: ").strip()
    if q.lower() == "exit":
        break
    if not q:
        continue
    results = search(q, top_k=5)
    print("\nTop matches for:", q)
    for text, src, score in results:
        print(f"\n[{src}] (score {score:.2f})")
        print(text)
