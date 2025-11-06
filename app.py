from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os, yaml

CONFIG = yaml.safe_load(open("config.yaml"))
VECTOR_DIR = CONFIG["vector_db_dir"]
LLM_MODEL = CONFIG["llm_model"]
EMB_MODEL = CONFIG["embedding_model"]

app = FastAPI(title="OCSF AI Search")

print("Loading embeddings and vectorstore...")
embeddings = SentenceTransformerEmbeddings(model_name=EMB_MODEL)
vectordb = Chroma(persist_directory=VECTOR_DIR, embedding_function=embeddings)

print("Loading Gemma model...")
# Read token from environment
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

MODEL_ID = "google/gemma-2b-it"  # or whichever Gemma model you’re using

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    torch_dtype="auto",
    device_map="auto"
)
print("Gemma ready.")

def retrieve_context(query, k=5):
    docs = vectordb.similarity_search(query, k=k)
    return "\n\n".join([f"[{d.metadata['source']}] {d.page_content}" for d in docs])

@app.post("/query")
async def query(prompt: str, max_tokens: int = 512):
    context = retrieve_context(prompt)
    full_prompt = f"""You are an expert on the Open Cybersecurity Schema Framework (OCSF).
Use the following context from official OCSF repositories to answer questions accurately.

Context:
{context}

Question: {prompt}
Answer:"""

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"answer": answer, "context_used": context[:1000] + "..."}
