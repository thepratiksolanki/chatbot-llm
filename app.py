import os
import pickle
from flask import Flask, request, jsonify, send_from_directory
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from flask_cors import CORS
from rapidfuzz import fuzz

# ---------- Step 1: Init ----------
app = Flask(__name__)
CORS(app)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")

# Local directory for vectorstores
VECTOR_DB_DIR = "vector_dbs"
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# ---------- Step 2: Helper Functions ----------
def get_tenant_db_path(tenant_id: str) -> str:
    return os.path.join(VECTOR_DB_DIR, f"{tenant_id}.pkl")

def load_vectorstore(tenant_id: str):
    path = get_tenant_db_path(tenant_id)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def save_vectorstore(tenant_id: str, vectorstore):
    path = get_tenant_db_path(tenant_id)
    with open(path, "wb") as f:
        pickle.dump(vectorstore, f)

# ---------- Step 3: Serve Frontend ----------
@app.route("/")
def index():
    # Serve index.html from root directory
    return send_from_directory(".", "index.html")

# ---------- Step 4: API to Upload KB ----------
@app.route("/upload", methods=["POST"])
def upload():
    data = request.json
    tenant_id = data.get("tenant_id")
    docs = data.get("docs", [])

    if not tenant_id or not docs:
        return jsonify({"error": "Missing tenant_id or docs"}), 400

    texts = [d["content"] for d in docs]
    metadatas = [{"title": d.get("title", "Untitled"), "url": d.get("url", "")} for d in docs]

    vectorstore = FAISS.from_texts(texts, embedding=embedding_model, metadatas=metadatas)
    save_vectorstore(tenant_id, vectorstore)

    return jsonify({"message": f"✅ KB uploaded for tenant {tenant_id}", "docs_added": len(docs)})

# ---------- Step 5: API to Search KB ----------
@app.route("/search", methods=["GET"])
def search():
    tenant_id = request.args.get("tenant_id")
    query = request.args.get("query", "").lower()

    if not tenant_id or not query:
        return jsonify({"error": "Missing tenant_id or query"}), 400

    vectorstore = load_vectorstore(tenant_id)
    if not vectorstore:
        return jsonify({"error": f"No KB found for tenant {tenant_id}"}), 404

    # --- Semantic search
    semantic_results = vectorstore.similarity_search_with_score(query, k=10)
    semantic_hits = [
        {
            "title": r.metadata.get("title", "Untitled"),
            "url": r.metadata.get("url", ""),
            "snippet": r.page_content[:200],
            "score": float(score),
            "source": "semantic"
        }
        for r, score in semantic_results
    ]

    # --- Fuzzy search
    all_docs = []
    for doc in vectorstore.docstore._dict.values():
        all_docs.append({
            "title": doc.metadata.get("title", "Untitled"),
            "url": doc.metadata.get("url", ""),
            "content": doc.page_content
        })

    fuzzy_hits = []
    for d in all_docs:
        score_title = fuzz.partial_ratio(query, d["title"].lower())
        score_content = fuzz.partial_ratio(query, d["content"].lower())
        score = max(score_title, score_content)

        if score >= 80:
            if score_title == 100:
                boosted_score = 200
            elif score_content == 100:
                boosted_score = 180
            elif score_title > score_content:
                boosted_score = score + 50
            else:
                boosted_score = score

            fuzzy_hits.append({
                "title": d["title"],
                "url": d["url"],
                "snippet": d["content"][:200],
                "score": boosted_score,
                "source": "fuzzy"
            })

    # Deduplicate & merge results
    seen_urls = set()
    final_results = []
    for item in sorted(fuzzy_hits, key=lambda x: x["score"], reverse=True):
        if len(final_results) >= 6:
            break
        if item["url"] not in seen_urls:
            final_results.append(item)
            seen_urls.add(item["url"])

    max_fuzzy_score = max([f['score'] for f in fuzzy_hits], default=0)
    if len(final_results) < 6 or max_fuzzy_score < 80:
        for item in sorted(semantic_hits, key=lambda x: x["score"], reverse=True):
            if len(final_results) >= 6:
                break
            if item["url"] not in seen_urls:
                final_results.append(item)
                seen_urls.add(item["url"])

    return jsonify(final_results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
