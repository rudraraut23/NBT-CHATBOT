# streamlit_app.py
import os
import shutil
import base64
import asyncio
import traceback
import json
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

import streamlit as st
from dotenv import load_dotenv
import nest_asyncio

# LangChain / model imports (match your environment)
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore

# LlamaParse (may be async or sync depending on version)
from llama_parse import LlamaParse

# For safe SBERT usage:
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # handled later

# allow asyncio.run inside Streamlit
nest_asyncio.apply()

# Streamlit rerun compatibility (old/new versions)
if not hasattr(st, "rerun") and hasattr(st, "experimental_rerun"):
    st.rerun = st.experimental_rerun

# Page config
st.set_page_config(page_title="NBT Advanced CHATBOT", page_icon="🤖", layout="wide")
st.markdown("<style>.main{background-color:#f7f9fc;}</style>", unsafe_allow_html=True)

# Helpers
# --- CORRECTED FUNCTION ---
# --- FULLY CORRECTED FUNCTION ---
def autoplay_audio(file_path: str):
    """
    Plays an audio file from a path by encoding it in base64.
    """
    # FIX: This now correctly checks the 'file_path' variable 
    # (e.g., "notification.mp3")
    if os.path.exists(file_path):
        try:
            # This correctly opens the 'file_path' in binary mode
            with open(file_path, "rb") as f:
                data = f.read()
            
            b64 = base64.b64encode(data).decode()
            md = f'<audio controls autoplay="true" style="display:none;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
            st.markdown(md, unsafe_allow_html=True)
        
        except Exception as e:
            st.warning(f"Error playing audio {file_path}: {e}")
    else:
        # This 'else' block now works correctly.
        st.warning(f"Audio file not found at: {file_path}")

def safe_rmtree(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path, ignore_errors=True)         
        except Exception:
            pass

# Async thread executor helpers
executor = ThreadPoolExecutor(max_workers=4)
async def run_blocking(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: fn(*args, **kwargs))

# Robust loader for parser.load_data that supports both sync and async variants
async def robust_load_data(parser, path):
    maybe = parser.load_data(path)
    if asyncio.iscoroutine(maybe):
        return asyncio.get_event_loop().run_until_complete(maybe)
    return maybe

# Load .env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llama_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

# Warn if keys missing
if not groq_api_key:
    st.warning("GROQ_API_KEY not found in .env. If you want hosted LLM answers use Groq. You can still parse/index documents.")

# --- SAFE embeddings initialization using sentence-transformers on CPU ---
if SentenceTransformer is None:
    st.error("Module 'sentence_transformers' not installed. Install with: pip install sentence-transformers")
    raise SystemExit("Install sentence-transformers and restart.")

# Load SBERT on CPU explicitly to avoid meta-tensor device moves
sbert = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

class SimpleEmbeddingWrapper:
    """
    Minimal wrapper exposing embed_documents(texts) and embed_query(text).
    Returns python lists of floats (suitable for most vector stores).
    """
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        # convert_to_numpy=True for faster results; returns np.ndarray
        embeddings_np = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [vec.tolist() for vec in embeddings_np]

    def embed_query(self, text):
        vec = self.model.encode([text], show_progress_bar=False, convert_to_numpy=True)
        return vec[0].tolist()

embeddings = SimpleEmbeddingWrapper(sbert)

# Session state defaults
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'processed_file_names' not in st.session_state:
    st.session_state.processed_file_names = []
if 'vectorstore_initialized' not in st.session_state:
    st.session_state.vectorstore_initialized = False

# Sidebar
with st.sidebar:
    st.header("Advanced Controls")
    if st.button("Clear Memory & Re-process"):
        st.session_state.chat_history = ChatMessageHistory()
        st.session_state.rag_chain = None
        st.session_state.processed_file_names = []
        st.session_state.vectorstore_initialized = False
        
        st.success("Memory wiped. Re-upload files to re-index.")
        st.rerun()

    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, DOCX, Images)",
        type=["pdf", "docx", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

# Title
st.title("🤖 NBT Advanced RAG Chatbot")


# === LLM Initialization (prefer Groq hosted API if key present) ===
llm = None
if groq_api_key:
    try:
        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
        
    except Exception as e:
        st.error(f"Failed to initialize ChatGroq: {e}")
        llm = None
else:
    st.info("No GROQ_API_KEY provided. LLM responses will not be generated until you configure an LLM (Groq or local).")

# === Ingestion & indexing (only when uploads change) ===
current_file_names = sorted([f.name for f in uploaded_files]) if uploaded_files else []

if uploaded_files and st.session_state.processed_file_names != current_file_names:
    temp_dir = "temp_files"
    safe_rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    with st.spinner("Parsing & indexing files..."):
        try:
            # Initialize parser (LlamaParse)
            parser = LlamaParse(
                api_key=llama_api_key,
                result_type="markdown",
                verbose=False,
                language="en"
            )

            parent_docs = []
            for uploaded_file in uploaded_files:
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                st.write(f"Parsing **{uploaded_file.name}**...")
                try:
                    maybe = parser.load_data(temp_path)
                    if asyncio.iscoroutine(maybe):
                        st.write("Awaiting async parser.load_data(...)")
                        llama_docs = asyncio.get_event_loop().run_until_complete(maybe)
                    else:
                        llama_docs = maybe

                    if not llama_docs:
                        st.warning(f"No parsed pages for {uploaded_file.name}.")
                        llama_docs = []

                    for idx, doc in enumerate(llama_docs):
                        text = getattr(doc, "text", None) or getattr(doc, "content", None) or ""
                        metadata = getattr(doc, "metadata", {}) or {}
                        page_label = metadata.get("page_label", idx + 1)
                        src_name = os.path.basename(uploaded_file.name)

                        lc_doc = Document(
                            page_content=text,
                            metadata={**metadata, "source": src_name, "page": page_label}
                        )
                        unique_id = f"{src_name}_p{page_label}"
                        lc_doc.metadata["doc_id"] = unique_id
                        parent_docs.append(lc_doc)

                except Exception as e_doc:
                    st.error(f"Failed to parse {uploaded_file.name}: {e_doc}")
                    with open("parse_errors.log", "a") as logf:
                        logf.write(f"Error parsing {uploaded_file.name}:\n{traceback.format_exc()}\n\n")

            if not parent_docs:
                st.warning("No parseable content found.")
            else:
                # persistent Chroma
                vectorstore = Chroma(
                    collection_name="advanced_rag",
                    
                    embedding_function=embeddings,  # our wrapper provides embed_documents & embed_query
                )

                # metadata store
                store = InMemoryStore()
                child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

                retriever = ParentDocumentRetriever(
                    vectorstore=vectorstore,
                    docstore=store,
                    child_splitter=child_splitter,
                    id_key="doc_id",
                )

                # add parent docs and explicit ids
                doc_ids = [doc.metadata["doc_id"] for doc in parent_docs]
                retriever.add_documents(parent_docs, ids=doc_ids)

                # Persist Chroma (if supported)
                

                # Build LLM chain (LangChain)
                if llm is None:
                    st.warning("No LLM configured — answers cannot be generated until an LLM is available (set GROQ_API_KEY).")
                # --- NEW, "LESS STRICT" PROMPT ---
                system_prompt = (
    "You are a helpful and conversational document assistant. "
    "Your goal is to answer the user's question using the provided context. "
    "Read the context carefully and find the most relevant information to form a helpful answer. "
    
    # This line encourages it to answer, even if it's not a perfect 1-to-1 match
    "Do your best to answer the user's question, even if the query is conversational "
    "or not a perfect match for the text. "
    
    # This gives it a "polite way out" instead of a hard failure
    "If the context is completely unrelated or does not contain the answer, "
    "just say: 'I've checked the documents, but I can't find a clear answer to that specific question.'\n\n"
    
    "Here is the context:\n{context}"
                )

                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])

                history_retriever = create_history_aware_retriever(
                    llm, retriever, ChatPromptTemplate.from_messages([
                        ("system", "Rephrase the user question to be standalone, referencing chat history if needed."),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ])
                )

                rag_chain = create_retrieval_chain(history_retriever, create_stuff_documents_chain(llm, qa_prompt))

                st.session_state.rag_chain = RunnableWithMessageHistory(
                    rag_chain,
                    lambda session_id: st.session_state.chat_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer",
                )

                st.session_state.processed_file_names = current_file_names
                st.session_state.vectorstore_initialized = True
                st.success(f"Indexed {len(parent_docs)} pages from {len(current_file_names)} files.")

        except Exception as e:
            st.error(f"Indexing failed: {e}")
            with open("parse_errors.log", "a") as logf:
                logf.write(f"Indexing pipeline exception:\n{traceback.format_exc()}\n\n")
        finally:
            safe_rmtree(temp_dir)

    # refresh UI after indexing
    st.rerun()

# --- Show existing chat history ---
for msg in st.session_state.chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

# --- Chat input and answering ---
if user_input := st.chat_input("Ask about your documents..."):
    st.chat_message("human").write(user_input)

    identity_triggers = ["who are you", "about yourself", "what are you", "your identity", "what do you do"]
    is_identity_question = any(trigger in user_input.lower() for trigger in identity_triggers)

    if st.session_state.rag_chain and st.session_state.vectorstore_initialized:
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": "default"}}
                )

                if isinstance(response, dict):
                    answer_text = response.get("answer") or "Based on the documents provided, I cannot answer this question."
                    raw_context = response.get("context")
                else:
                    answer_text = str(response) or "Based on the documents provided, I cannot answer this question."
                    raw_context = None

                st.chat_message("ai").write(answer_text)
                try:
                    st.session_state.chat_history.add_user_message(user_input)
                    st.session_state.chat_history.add_ai_message(answer_text)
                except Exception:
                    pass

                # show sources defensively
                sources = set()
                if raw_context:
                    for d in raw_context:
                        try:
                            md = getattr(d, "metadata", {}) if d else {}
                            src = md.get("source") or md.get("filename")
                            page = md.get("page") or md.get("page_label")
                            if src:
                                sources.add(f"{src} (Page {page})" if page else f"{src}")
                        except Exception:
                            continue
                if sources:
                    with st.expander("Sources Used"):
                        for s in sorted(sources):
                            st.write(f"📄 {s}")

                if os.path.exists("notification.mp3"):
                    autoplay_audio("notification.mp3")

            except Exception as e:
                st.error(f"Query error: {e}")
                with open("parse_errors.log", "a") as logf:
                    logf.write(f"Query error:\n{traceback.format_exc()}\n\n")

    elif is_identity_question:
        ai_response = "I am NBT CHATBOT. I answer questions using uploaded documents."
        st.chat_message("ai").write(ai_response)
        try:
            st.session_state.chat_history.add_user_message(user_input)
            st.session_state.chat_history.add_ai_message(ai_response)
        except Exception:
            pass
    else:
        st.warning("Please upload documents first and wait for indexing to finish so I can answer.")
