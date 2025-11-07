import streamlit as st
import os
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import torch

# Page config
st.set_page_config(
    page_title="🔮 OCSF AI - Crystal Ball",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration


@st.cache_data
def load_config():
    return yaml.safe_load(open("config.yaml"))


CONFIG = load_config()
VECTOR_DIR = CONFIG["vector_db_dir"]
LLM_MODEL = CONFIG["llm_model"]
EMB_MODEL = CONFIG["embedding_model"]

# Initialize embeddings and vectorstore


@st.cache_resource
def load_vectorstore():
    print("Loading embeddings and vectorstore...")
    embeddings = SentenceTransformerEmbeddings(model_name=EMB_MODEL)
    vectordb = Chroma(persist_directory=VECTOR_DIR,
                      embedding_function=embeddings)
    return vectordb


@st.cache_resource
def load_model():
    print("Loading Gemma model...")
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    MODEL_ID = LLM_MODEL

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Gemma ready.")
    return tokenizer, model


def retrieve_context(query, vectordb, k=5):
    """Retrieve relevant context from the vector database"""
    docs = vectordb.similarity_search(query, k=k)
    context_text = "\n\n".join(
        [f"[{d.metadata['source']}] {d.page_content}" for d in docs])
    sources = [d.metadata['source'] for d in docs]
    return context_text, sources


def build_conversation_prompt(messages, current_query, context_text):
    """Build a prompt that includes conversation history"""
    conversation_history = ""
    for msg in messages[-6:]:  # Keep last 6 messages for context
        if msg["role"] == "user":
            conversation_history += f"Human: {msg['content']}\n"
        else:
            conversation_history += f"Assistant: {msg['content']}\n"

    prompt = f"""You are an expert on the Open Cybersecurity Schema Framework (OCSF).
You are having a conversation with a user about OCSF. Use the provided context and conversation history to answer questions accurately and helpfully.

Context from OCSF documentation:
{context_text}

Conversation history:
{conversation_history}

Current question: {current_query}

Provide a helpful and accurate response based on the context and conversation history. Keep your response concise but informative."""

    return prompt


def generate_response(query, messages, tokenizer, model, vectordb, max_tokens=512):
    """Generate a response using the LLM with conversation context"""

    # Retrieve relevant context
    context_text, sources = retrieve_context(query, vectordb)

    # Build prompt with conversation history
    full_prompt = build_conversation_prompt(messages, query, context_text)

    # Generate response
    inputs = tokenizer(full_prompt, return_tensors="pt",
                       truncation=True, max_length=2048).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the generated tokens
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(
        generated_tokens, skip_special_tokens=True).strip()

    return response, sources


def main():
    # Load resources
    try:
        vectordb = load_vectorstore()
        tokenizer, model = load_model()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

    # Header
    st.title("🔮 OCSF AI - Crystal Ball")
    st.markdown("*Intelligent security data analysis with AI-powered insights*")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        max_tokens = st.slider("Max Response Length", 100, 1000, 512, 50)
        context_docs = st.slider("Context Documents", 1, 10, 5)

        st.header("📚 About")
        st.markdown("""
        This AI assistant helps you explore and understand the Open Cybersecurity Schema Framework (OCSF).
        
        **Features:**
        - Multi-turn conversations
        - Context-aware responses
        - Source citations
        - OCSF expertise
        """)

        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources", expanded=False):
                    for source in set(message["sources"]):  # Remove duplicates
                        st.text(f"• {source}")

    # Chat input
    if prompt := st.chat_input("Ask me about OCSF..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response, sources = generate_response(
                        prompt,
                        # Exclude current message
                        st.session_state.messages[:-1],
                        tokenizer,
                        model,
                        vectordb,
                        max_tokens
                    )

                    st.markdown(response)

                    # Show sources
                    if sources:
                        with st.expander("📚 Sources", expanded=False):
                            for source in set(sources):
                                st.text(f"• {source}")

                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })

                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":
    main()
