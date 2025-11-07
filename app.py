from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os
from pydantic import BaseModel
import yaml

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


# --- Change the retrieve_context function ---
def retrieve_context(query, k=5):
    docs = vectordb.similarity_search(query, k=k)
    context_text = "\n\n".join([f"[{d.metadata['source']}] {d.page_content}" for d in docs])
    sources = [d.metadata['source'] for d in docs]  # just the filenames
    return context_text, sources


class QueryRequest(BaseModel):
    prompt: str
    max_tokens: int = 512


@app.post("/query")
async def query(request: QueryRequest):
    prompt = request.prompt
    max_tokens = request.max_tokens

    context_text, sources = retrieve_context(prompt)

    full_prompt = f"""You are an expert on the Open Cybersecurity Schema Framework (OCSF).
Use the context provided to answer the question concisely in 1-2 sentences.

Context:
{context_text}

Question: {prompt}
Answer:"""

    # Tokenize input and generate
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)

    # Only decode the newly generated tokens (exclude prompt)
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return {
        "question": prompt,
        "answer": answer,
        "sources": sources
    }
