import os
import yaml
import json
from pathlib import Path
from git import Repo
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import markdown

CONFIG = yaml.safe_load(open("config.yaml"))

REPOS = CONFIG["repos"]
DATA_DIR = Path(CONFIG["local_data_dir"])
VECTOR_DIR = CONFIG["vector_db_dir"]
EMB_MODEL = CONFIG["embedding_model"]


def clone_repos():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for repo_url in REPOS:
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        dest = DATA_DIR / repo_name
        if not dest.exists():
            print(f"Cloning {repo_url}...")
            Repo.clone_from(repo_url, dest)
        else:
            print(f"Updating {repo_name}...")
            repo = Repo(dest)
            repo.remotes.origin.pull()


def load_files():
    """Collect all markdown and JSON files"""
    files = list(DATA_DIR.rglob("*.md")) + list(DATA_DIR.rglob("*.json"))
    print(f"Found {len(files)} files.")
    docs = []
    for f in files:
        try:
            if f.suffix == ".md":
                text = open(f, encoding="utf-8").read()
            elif f.suffix == ".json":
                with open(f, encoding="utf-8") as jf:
                    try:
                        data = json.load(jf)
                        text = json.dumps(data, indent=2)
                    except:
                        text = jf.read()
            else:
                continue
            docs.append({"text": text, "path": str(f)})
        except Exception as e:
            print(f"Skipping {f}: {e}")
    return docs


def build_vector_db(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["chunk_size"], chunk_overlap=CONFIG["chunk_overlap"]
    )
    texts, metas = [], []
    for d in docs:
        chunks = splitter.split_text(d["text"])
        for c in chunks:
            texts.append(c)
            metas.append({"source": d["path"]})
    print(f"Embedding {len(texts)} chunks...")
    embeddings = SentenceTransformerEmbeddings(model_name=EMB_MODEL)
    vectordb = Chroma.from_texts(
        texts, embeddings, metadatas=metas, persist_directory=VECTOR_DIR)
    vectordb.persist()
    print("Vector database built successfully!")


if __name__ == "__main__":
    clone_repos()
    docs = load_files()
    build_vector_db(docs)
