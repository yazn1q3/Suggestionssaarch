from flask import Flask, request, jsonify
from rapidfuzz import process
from txtai.embeddings import Embeddings
import json
import os

app = Flask(__name__)

# 🛒 تحميل المنتجات (ممكن تخليها DB)
with open("products.json", "r", encoding="utf-8") as f:
    products = json.load(f)

# 🧠 نظام ذكاء نصي (txtai يعمل Embeddings + Semantic Search)
embeddings = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})

# بناء الفهرس
for i, p in enumerate(products):
    embeddings.add(i, p["title"], None)
embeddings.save("index")

# 🔥 ملف لتعلم من المستخدم
SEARCHES_FILE = "searches.json"
if not os.path.exists(SEARCHES_FILE):
    with open(SEARCHES_FILE, "w", encoding="utf-8") as f:
        json.dump({}, f)

def save_search(term):
    """يسجل البحث في ملف JSON"""
    with open(SEARCHES_FILE, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data[term] = data.get(term, 0) + 1
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.truncate()

def get_popular():
    """يجيب أكتر الكلمات اللي الناس بتدور عليها"""
    with open(SEARCHES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        return sorted(data, key=data.get, reverse=True)[:5]

def smart_suggestions(query, top_k=7):
    # Semantic Search
    semantic = [p["text"] for p in embeddings.search(query, top_k)]

    # Fuzzy Search
    fuzzy = [r[0] for r in process.extract(query, [p["title"] for p in products], limit=top_k)]

    # Popular Searches
    popular = get_popular()

    # دمج الكل
    final = list(dict.fromkeys(semantic + fuzzy + popular))
    return final[:top_k]

@app.route("/api/suggestions", methods=["GET"])
def get_suggestions():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])

    # حفظ البحث عشان الخدمة تتعلم
    save_search(q)

    results = smart_suggestions(q)
    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
