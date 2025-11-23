from flask import Flask, render_template, request, jsonify
import os
import re
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

PDF_DIR = "pdfs"
STOP_PHRASES = ["p.t.o", "pto", "please turn over"]

QUESTION_START_RE = re.compile(
    r"""^(q[\.\s]*\d+|question\s*\d+|\d+\.\s*|\d+\)\s*[a-z]*|[a-h]\)\s*)""",
    re.IGNORECASE
)

def is_garbage_line(line):
    t = line.strip().lower()
    if not t:
        return True
    for phrase in STOP_PHRASES:
        if phrase in t:
            return True
    return False

def is_question_start(line):
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
                i = 0
                n = len(lines)
                while i < n:
                    line = lines[i]
                    if is_question_start(line) or i == 0:
                        block_lines = [line]
                        j = i + 1
                        while j < n and not is_question_start(lines[j]):
                            block_lines.append(lines[j])
                            j += 1
                        block = " ".join(block_lines)
                        if len(block) >= 30:
                            chunks.append(block)
                            sources.append(f"{fname} - page {page_index + 1}")
                        i = j
                    else:
                        i += 1
    return chunks, sources

chunks, sources = load_pdfs_to_chunks(PDF_DIR)
if not chunks:
    raise SystemExit("No text found in PDFs in 'pdfs' folder")

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
        results.append(
            {
                "text": chunks[global_idx],
                "source": sources[global_idx],
                "score": float(sims[local_idx])
            }
        )
    return results

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("message", "").strip()
    if not query:
        return jsonify({"answers": []})
    answers = search(query, top_k=5)
    return jsonify({"answers": answers})

if __name__ == "__main__":
    app.run(debug=True)
