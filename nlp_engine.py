import sqlite3
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'Database', 'alh_errors.db')
MODEL_PATH = os.path.join(BASE_DIR, 'Requirements', 'models', 'all-MiniLM-L6-v2')
CACHE_PATH = os.path.join(BASE_DIR, 'Database', 'vectors.pkl')

def load_model():
    print("Loading model...")
    model = SentenceTransformer(MODEL_PATH)
    print("Model loaded.")
    return model

def get_all_records():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT Error_ID, Component_Ref, Error_Type, Description, Corrective_Action FROM assembly_errors")
    rows = cursor.fetchall()
    conn.close()
    return rows

def build_vector_cache(model):
    print("Building vector cache...")
    records = get_all_records()
    descriptions = [row[3] for row in records]
    vectors = model.encode(descriptions, show_progress_bar=True)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump((records, vectors), f)
    print(f"Cache built and saved with {len(records)} vectors.")
    return records, vectors

def load_vector_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'rb') as f:
            records, vectors = pickle.load(f)
        print(f"Cache loaded with {len(records)} vectors.")
        return records, vectors
    return None, None

def search(query, model, records, vectors, top_k=3):
    query_vector = model.encode([query])
    scores = cosine_similarity(query_vector, vectors)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        record = records[idx]
        results.append({
            'Error_ID': record[0],
            'Component_Ref': record[1],
            'Error_Type': record[2],
            'Description': record[3],
            'Corrective_Action': record[4],
            'Match_Score': round(float(scores[idx]) * 100, 1)
        })
    return results

if __name__ == "__main__":
    models = load_model()
    records, vectors = load_vector_cache()
    if records is None:
        records, vectors = build_vector_cache(models)

    print("\n--- NLP ENGINE TEST ---\n")
    test_queries = [
        "connector mated to wrong port on avionics panel",
        "bolt not tightened enough on gearbox",
        "wire connected with wrong polarity on pump motor",
        "clamp missing on fuel line near firewall",
    ]
    for query in test_queries:
        print(f"Query: {query}")
        results = search(query, models, records, vectors)
        for i, r in enumerate(results):
            print(f"  Match {i+1} ({r['Match_Score']}%) — {r['Component_Ref']} | {r['Error_Type']}")
            print(f"  Fix: {r['Corrective_Action'][:80]}...")
        print()