import json, re, os, pickle, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoTokenizer as GenAutoTokenizer
import openai
import os
openai.api_key = os.getenv("OPENAI_API_KEY")


# --- Load & Clean Data ---
DATA_PATH = "dbcontent.txt"
with open(DATA_PATH, "r") as f:
    raw = f.read()
cleaned = re.sub(r'ObjectId\("([^"]+)"\)', r'"\1"', raw)
data = json.loads(cleaned)
documents = [
    {
        "transcript": doc["transcribetext"],
        "coursename": doc.get("coursename", "Unknown Course"),
        "videoname": doc.get("videoname", "Unknown Video"),
        "videourl": doc.get("videourl", "No URL")
    }
    for doc in data if "transcribetext" in doc
]

# --- Chunking ---
def chunk_text(text, tokenizer, max_tokens=512):
    tokens = tokenizer.encode(text)
    return [
        tokenizer.decode(tokens[i:i+max_tokens], skip_special_tokens=True)
        for i in range(0, len(tokens), max_tokens)
    ]

sbert_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
all_chunks = []
for doc in documents:
    for chunk in chunk_text(doc["transcript"], sbert_tokenizer, max_tokens=512):
        all_chunks.append({
            "chunk": chunk,
            "coursename": doc["coursename"],
            "videoname": doc["videoname"],
            "videourl": doc["videourl"]
        })

# --- TF-IDF Caching & Index ---
chunk_texts = [item["chunk"] for item in all_chunks]
vectorizer_cache = "vectorizer.pkl"
tfidf_cache = "tfidf_matrix.pkl"
if os.path.exists(vectorizer_cache) and os.path.exists(tfidf_cache):
    with open(vectorizer_cache, "rb") as f:
        vectorizer = pickle.load(f)
    with open(tfidf_cache, "rb") as f:
        tfidf_matrix = pickle.load(f)
else:
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(chunk_texts)
    with open(vectorizer_cache, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(tfidf_cache, "wb") as f:
        pickle.dump(tfidf_matrix, f)

def get_relevant_chunk(query):
    query_vec = vectorizer.transform([query])
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix)
    idx = np.argmax(cosine_sim)
    return all_chunks[idx]

# --- Generation using DialoGPT ---
gen_tokenizer = GenAutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
gen_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
def generate_answer(prompt):
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = gen_model.generate(inputs["input_ids"], max_new_tokens=100, pad_token_id=gen_tokenizer.eos_token_id)
    return gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Refinement via ChatGPT ---
def refine_answer(initial_answer, query, sources_text):
    refinement_prompt = (
        "You are an expert editor. Rewrite the answer below so that it is clear, grammatically correct, "
        "and directly addresses the question in under 100 words. Include a citation line with the course name, "
        "video name, and video URL at the end.\n\n"
        f"Question: {query}\n\n"
        f"Initial Answer: {initial_answer}\n\n"
        f"Sources: {sources_text}\n\n"
        "Refined Answer:"
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a skilled editor."},
            {"role": "user", "content": refinement_prompt}
        ],
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message["content"]

def chatbot_response(query):
    relevant = get_relevant_chunk(query)
    context = relevant["chunk"]
    sources_text = f"Course: {relevant['coursename']}, Video: {relevant['videoname']} - {relevant['videourl']}"
    prompt = (
        "Using only the context below, generate a concise, direct answer in under 100 words that addresses the question. "
        "Summarize and rephrase the relevant information in your own words.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    initial_answer = generate_answer(prompt)
    refined = refine_answer(initial_answer, query, sources_text)
    return refined
