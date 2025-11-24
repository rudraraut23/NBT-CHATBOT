# app.py
import base64
import os
import io
import time
import json
import hashlib
import shutil
import traceback
import re
from multiprocessing import cpu_count
from typing import List, Tuple, Sequence

import streamlit as st
from dotenv import load_dotenv
import nest_asyncio

# optional libraries
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None

# sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# langchain pieces
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# attempt chroma then fallback to FAISS
try:
    from langchain_chroma import Chroma
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False

try:
    from langchain.vectorstores import FAISS
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

from llama_parse import LlamaParse
from langchain_groq import ChatGroq

nest_asyncio.apply()

# ---------- Config ----------
UPLOADS_DIR = "uploaded_uploads"
TEMP_DIR = "tmp_parse"
HASH_STORE = "indexed_hashes.json"
FAISS_DIR = "faiss_index"
CHROMA_COLLECTION = "advanced_rag"

WORKERS = max(1, cpu_count() - 1)
CHUNK_SIZE_DEFAULT = 800
CHUNK_OVERLAP_DEFAULT = 200
EMBED_BATCH_SIZE = 128
MIN_TEXT_LEN = 30
RETRIEVE_K_DEFAULT = 5

DEFAULT_MODEL_NAME = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.0

# Developer-provided uploaded image path (will be transformed to URL by tooling if needed)
UPLOADED_IMAGE_PATH = "/mnt/data/50d3186c-4849-4057-90b0-ccf66cd970ac.png"

# ---------- Utilities ----------
def ensure_dirs():
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(FAISS_DIR, exist_ok=True)

def safe_rmtree(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass

def md5_bytes(b: bytes) -> str:
    m = hashlib.md5()
    m.update(b)
    return m.hexdigest()

def load_indexed_hashes() -> dict:
    try:
        with open(HASH_STORE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_indexed_hashes(d: dict):
    with open(HASH_STORE, "w") as f:
        json.dump(d, f)

# ---------- Parsing helpers ----------
def extract_pdf_pages(path: str) -> List[Tuple[int, str]]:
    res = []
    if fitz is None:
        return res
    try:
        doc = fitz.open(path)
        for i in range(len(doc)):
            try:
                page = doc.load_page(i)
                txt = page.get_text("text") or ""
            except Exception:
                txt = ""
            res.append((i + 1, txt))
        doc.close()
    except Exception:
        pass
    return res

def ocr_image_to_text(path: str) -> str:
    if pytesseract is None or Image is None:
        return ""
    try:
        img = Image.open(path)
        txt = pytesseract.image_to_string(img)
        return txt
    except Exception:
        return ""

def parse_file(path: str, filename: str, llama_api_key: str) -> List[Document]:
    docs = []
    lower = filename.lower()
    if lower.endswith(".pdf") and fitz is not None:
        pages = extract_pdf_pages(path)
        for pnum, text in pages:
            if text and len(text.strip()) >= MIN_TEXT_LEN:
                meta = {"source": filename, "page": pnum}
                docs.append(Document(page_content=text, metadata=meta))
        return docs

    if lower.endswith((".png", ".jpg", ".jpeg")):
        txt = ocr_image_to_text(path)
        if txt and len(txt.strip()) >= MIN_TEXT_LEN:
            meta = {"source": filename, "page": 1}
            docs.append(Document(page_content=txt, metadata=meta))
            return docs

    # fallback to LlamaParse for docx or other types
    try:
        parser = LlamaParse(api_key=llama_api_key, result_type="markdown", verbose=False, language="en")
        maybe = parser.load_data(path)
        if hasattr(maybe, "__iter__"):
            for idx, o in enumerate(maybe):
                text = getattr(o, "text", None) or getattr(o, "content", None) or ""
                if text and len(text.strip()) >= MIN_TEXT_LEN:
                    meta = getattr(o, "metadata", {}) or {}
                    page_lbl = meta.get("page_label", idx + 1)
                    meta_out = {"source": filename, "page": page_lbl, **meta}
                    docs.append(Document(page_content=text, metadata=meta_out))
    except Exception:
        pass
    return docs

# ---------- Embedding wrapper ----------
class SimpleEmbeddings:
    def __init__(self, model):
        self.model = model
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        arr = self.model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True, batch_size=EMBED_BATCH_SIZE)
        return [a.tolist() for a in arr]
    def embed_query(self, text: str) -> List[float]:
        arr = self.model.encode([text], show_progress_bar=False, convert_to_numpy=True)
        return arr[0].tolist()

# ---------- Vectorstore helpers ----------
def init_vectorstore_chroma(embeddings):
    if not CHROMA_AVAILABLE:
        raise RuntimeError("Chroma is not available")
    vs = Chroma(collection_name=CHROMA_COLLECTION, embedding_function=embeddings)
    return vs

def init_vectorstore_faiss_from_texts(texts, metas, embeddings):
    if not FAISS_AVAILABLE:
        raise RuntimeError("FAISS not available")
    return FAISS.from_texts(texts, embeddings, metadatas=metas)

# ---------- Notification sound ----------
def play_notification_sound():
    sound_file = "notification.mp3"  # adjust if your filename is different

    if os.path.exists(sound_file):
        with open(sound_file, "rb") as f:
            audio_bytes = f.read()

        encoded = base64.b64encode(audio_bytes).decode()

        audio_html = f"""
        <audio autoplay="true" style="display:none;">
            <source src="data:audio/mp3;base64,{encoded}" type="audio/mp3">
        </audio>
        """

        st.markdown(audio_html, unsafe_allow_html=True)
    else:
        # do not spam warning if sound missing
        pass

# ---------- Small verification helpers ----------
def extract_numbers(s: str):
    return re.findall(r"\d+(?:\.\d+)?", s)

def ngram_overlap(a: str, b: str, n: int = 5) -> float:
    atoks = [t for t in re.findall(r"\w+", a.lower())]
    btoks = [t for t in re.findall(r"\w+", b.lower())]
    if len(atoks) < n:
        return 0.0
    a_ngrams = {" ".join(atoks[i:i+n]) for i in range(len(atoks)-n+1)}
    b_text = " ".join(btoks)
    matches = sum(1 for ng in a_ngrams if ng in b_text)
    return matches / max(1, len(a_ngrams))

# ---------- App UI & Flow ----------
load_dotenv()
ensure_dirs()

st.set_page_config(page_title="NBT Advanced RAG (Optimized)", page_icon="🤖", layout="wide")
st.markdown("<style>.main{background-color:#f7f9fc;}</style>", unsafe_allow_html=True)

# Keys
groq_api_key = os.getenv("GROQ_API_KEY")
llama_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

# Use defaults (removed left sidebar controls as requested)
CHUNK_SIZE = CHUNK_SIZE_DEFAULT
CHUNK_OVERLAP = CHUNK_OVERLAP_DEFAULT
RETRIEVE_K = RETRIEVE_K_DEFAULT

# ---------- Replace uploader + clear logic with this block ----------
# make sure UPLOADS_DIR, HASH_STORE, FAISS_DIR and helper functions (ensure_dirs, safe_rmtree, load_indexed_hashes, save_indexed_hashes) exist above

# Use a key so we can reset the uploader widget
# ---------- Safe uploader + clear logic (avoid modifying widget-backed keys) ----------
# ---------------- FIXED Uploader + Clear Logic ----------------
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = "uploader_1"

def hard_reset_uploader():
    # cycle to a brand-new key → forces Streamlit to recreate the uploader widget
    st.session_state.uploader_key = f"uploader_{time.time()}"

    # remove uploaded files folder
    if os.path.exists(UPLOADS_DIR):
        for f in os.listdir(UPLOADS_DIR):
            try:
                os.remove(os.path.join(UPLOADS_DIR, f))
            except:
                pass

    # delete index data
    if os.path.exists(HASH_STORE):
        try: os.remove(HASH_STORE)
        except: pass

    safe_rmtree(FAISS_DIR)

    # clear session state objects
    for k in [
        "processed_file_names", "vectorstore", "vectorstore_kind", "rag_chain",
        "file_hashes", "file_paths", "chat_history"
    ]:
        if k in st.session_state:
            del st.session_state[k]

    st.session_state.processed_file_names = []
    st.session_state.file_hashes = {}
    st.session_state.file_paths = {}

    st.rerun()


uploaded_files = st.file_uploader(
    "Upload Documents (PDF, DOCX, Images)",
    accept_multiple_files=True,
    type=["pdf", "docx", "png", "jpg", "jpeg"],
    key=st.session_state.uploader_key
)

if st.button("Clear uploaded files and indexing"):
    hard_reset_uploader()
# ---------------- END FIXED BLOCK ----------------

if SentenceTransformer is None:
    st.error("Install sentence-transformers to enable embeddings: pip install sentence-transformers")
    st.stop()

# initialize SBERT
import torch
use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
if use_cuda:
    st.info("GPU available — embeddings will run on GPU.")
else:
    st.info("NBT AT CPU")

sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
embeddings_wrapper = SimpleEmbeddings(sbert_model)

# session defaults
if 'processed_file_names' not in st.session_state:
    st.session_state.processed_file_names = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'file_hashes' not in st.session_state:
    st.session_state.file_hashes = load_indexed_hashes()
if 'file_paths' not in st.session_state:
    st.session_state.file_paths = {}

st.title("🤖 NBT Advanced RAG Chatbot ")

# Initialize LLM (Groq) if available
llm = None
if groq_api_key:
    try:
        llm = ChatGroq(model_name=DEFAULT_MODEL_NAME, temperature=LLM_TEMPERATURE)
    except Exception:
        st.warning("Could not initialize Groq LLM; continue with indexing only.")

# Indexing when uploads changed
current_names = sorted([f.name for f in uploaded_files]) if uploaded_files else []
if uploaded_files and st.session_state.processed_file_names != current_names:
    # persist uploads, compute MD5 and decide which to process
    ensure_dirs()
    to_process = []
    for uf in uploaded_files:
        content = uf.getvalue()
        h = md5_bytes(content)
        dest = os.path.join(UPLOADS_DIR, uf.name)
        try:
            with open(dest, "wb") as wf:
                wf.write(content)
        except Exception:
            st.error(f"Cannot write uploaded file to disk: {uf.name}. Check server permission.")
            continue
        st.session_state.file_paths[uf.name] = dest
        prev_h = st.session_state.file_hashes.get(uf.name)
        if prev_h == h and st.session_state.vectorstore is not None:
            st.info(f"Skipping unchanged file: {uf.name}")
            continue
        to_process.append((uf.name, dest, h))

    if not to_process:
        st.success("No new/changed files to index.")
        st.session_state.processed_file_names = current_names
    else:
        # parse sequentially for visible progress
        parse_bar = st.progress(0)
        parsed_docs = []
        total = len(to_process)
        i = 0
        for name, path, h in to_process:
            st.write(f"Parsing {name} ...")
            docs = parse_file(path, name, llama_api_key)
            parsed_docs.extend(docs)
            i += 1
            parse_bar.progress(i / total)
        parse_bar.empty()

        if not parsed_docs:
            st.warning("No parseable content found.")
        else:
            st.info(f"Parsed {len(parsed_docs)} parent pages. Chunking...")

            splitter = RecursiveCharacterTextSplitter(chunk_size=int(CHUNK_SIZE), chunk_overlap=int(CHUNK_OVERLAP))
            child_docs = []
            total_parents = len(parsed_docs)
            p = 0
            chunk_bar = st.progress(0)
            for doc in parsed_docs:
                chunks = splitter.split_documents([doc])
                for c in chunks:
                    text = c.page_content or ""
                    if len(text.strip()) < MIN_TEXT_LEN:
                        continue
                    meta = c.metadata or {}
                    src = meta.get("source", doc.metadata.get("source"))
                    page = meta.get("page", doc.metadata.get("page"))
                    chunk_id = f"{src}_p{page}_{hashlib.md5(text.encode()).hexdigest()[:8]}"
                    c.metadata = {"source": src, "page": page, "chunk_id": chunk_id}
                    child_docs.append(c)
                p += 1
                chunk_bar.progress(p / total_parents)
            chunk_bar.empty()

            st.info(f"Created {len(child_docs)} chunks. Embedding & indexing...")

            texts = [c.page_content for c in child_docs]
            metadatas = [c.metadata for c in child_docs]

            # embedding & vectorstore init
            def embed_and_index():
                try:
                    if CHROMA_AVAILABLE:
                        st.info("Initializing Chroma local vectorstore.")
                        vs = init_vectorstore_chroma(embeddings_wrapper)
                        # add texts (some wrappers provide add_texts)
                        try:
                            vs.add_texts(texts, metadatas=metadatas)
                        except Exception:
                            # fallback: try from_texts
                            try:
                                vs = init_vectorstore_chroma(embeddings_wrapper)
                                vs.from_texts(texts, embeddings_wrapper, metadatas=metadatas)
                            except Exception:
                                raise
                        return ("chroma", vs)
                    else:
                        raise RuntimeError("Chroma not available")
                except Exception as exc_chroma:
                    st.warning(f"Chroma init failed — falling back to FAISS: {exc_chroma}")
                    try:
                        if not FAISS_AVAILABLE:
                            raise RuntimeError("FAISS not available")
                        vs = init_vectorstore_faiss_from_texts(texts, metadatas, embeddings_wrapper)
                        try:
                            vs.save_local(FAISS_DIR)
                        except Exception:
                            pass
                        return ("faiss", vs)
                    except Exception as exc_faiss:
                        st.error(f"Failed to init vectorstore: {exc_faiss}")
                        raise

            try:
                kind, vectorstore = embed_and_index()
                st.session_state.vectorstore = vectorstore
                st.session_state.vectorstore_kind = kind
                for name, path, h in to_process:
                    st.session_state.file_hashes[name] = h
                save_indexed_hashes(st.session_state.file_hashes)
                st.success(f"Indexing complete ({kind}).")
            except Exception:
                st.error("Indexing failed, check parse_errors.log")
                with open("parse_errors.log", "a") as lf:
                    lf.write(traceback.format_exc())

    st.session_state.processed_file_names = current_names
    time.sleep(0.2)
    st.rerun()

# show chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()
for msg in st.session_state.chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

# chat input
if user_input := st.chat_input("Ask about your documents (legal/CA friendly)..."):
    st.chat_message("human").write(user_input)

    # identity question
    identity_triggers = ["who are you", "about yourself", "what are you", "your identity", "what do you do"]
    is_identity = any(t in user_input.lower() for t in identity_triggers)
    handled = False

    if is_identity:
        ai_resp = "I am NBT Advanced RAG — I answer questions using your uploaded documents (CA / legal friendly)."
        st.chat_message("ai").write(ai_resp)
        try:
            st.session_state.chat_history.add_user_message(user_input)
            st.session_state.chat_history.add_ai_message(ai_resp)
        except Exception:
            pass
        handled = True

    # If not handled and no vectorstore -> warn
    if not handled and st.session_state.vectorstore is None:
        st.warning("Please upload and index documents first.")
        handled = True

    # If we already handled response (identity or no vectorstore), skip retrieval
    if not handled:
        # perform retrieval + RAG
        vs = st.session_state.vectorstore
        try:
            if getattr(vs, "as_retriever", None):
                retriever = vs.as_retriever(search_kwargs={"k": int(RETRIEVE_K)})
            else:
                retriever = vs.get_retriever(k=int(RETRIEVE_K))
        except Exception:
            class _SimpleRetriever:
                def __init__(self, vs):
                    self.vs = vs
                def get_relevant_documents(self, query, k=RETRIEVE_K):
                    try:
                        return self.vs.similarity_search(query, k=k)
                    except Exception:
                        return []
            retriever = _SimpleRetriever(vs)

        with st.spinner("Retrieving relevant passages..."):
            try:
                docs = retriever.get_relevant_documents(user_input)
            except Exception:
                try:
                    docs = vs.similarity_search(user_input, k=int(RETRIEVE_K))
                except Exception:
                    docs = []

        if not docs:
            ai_resp = "I've checked the documents, but I can't find a clear answer to that specific question."
            st.chat_message("ai").write(ai_resp)
            try:
                st.session_state.chat_history.add_user_message(user_input)
                st.session_state.chat_history.add_ai_message(ai_resp)
            except Exception:
                pass
            # do not show pages
        else:
            context_pieces = []
            for d in docs:
                md = getattr(d, "metadata", {}) or {}
                src = md.get("source") or md.get("filename") or "source"
                page = md.get("page") or md.get("page_label") or ""
                header = f"{src} (Page {page})" if page else src
                snippet = (d.page_content or "").strip()
                snippet_short = snippet[:1200] + (" …" if len(snippet) > 1200 else "")
                context_pieces.append(f"---\nSource: {header}\n{snippet_short}")
            full_context = "\n\n".join(context_pieces)

            system_prompt = (
                "You are a helpful document assistant specialized for legal / chartered-accountant documents. "
                "Your first priority is to answer questions using the provided document context. "
                "Cite the document sources and page numbers you used (in parentheses) after factual statements or quotes. "
                "If the documents provide the answer, respond concisely and include source citations. "
                "If the documents do not contain an answer, say: "
                "'I've checked the documents, but I can't find a clear answer to that specific question.' "
                "You may provide brief clarifications or next steps, but do NOT hallucinate facts not supported by the documents unless explicitly asked for background."
                "\n\nContext:\n{context}"
            )

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])

            try:
                history_retriever = create_history_aware_retriever(
                    llm, retriever, ChatPromptTemplate.from_messages([
                        ("system", "Rephrase the user question to be standalone, referencing chat history if needed."),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ])
                )
                rag_chain = create_retrieval_chain(history_retriever, create_stuff_documents_chain(llm, qa_prompt))
                runnable = RunnableWithMessageHistory(
                    rag_chain,
                    lambda session_id: st.session_state.chat_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer",
                )

                with st.spinner("Thinking (using retrieved context)..."):
                    response = runnable.invoke({"input": user_input, "context": full_context},
                                               config={"configurable": {"session_id": "default"}})
            except Exception:
                st.error("RAG chain error; see parse_errors.log")
                with open("parse_errors.log", "a") as lf:
                    lf.write(traceback.format_exc())
                response = None

            if isinstance(response, dict):
                answer_text = response.get("answer") or "Based on the documents provided, I cannot answer this question."
            else:
                answer_text = str(response) if response else "Based on the documents provided, I cannot answer this question."

            # show assistant answer
            st.chat_message("ai").write(answer_text)
            play_notification_sound()
            try:
                st.session_state.chat_history.add_user_message(user_input)
                st.session_state.chat_history.add_ai_message(answer_text)
            except Exception:
                pass



if "answer_text" in locals():
    refusal_str = "I've checked the documents, but I can't find a clear answer to that specific question."
    if refusal_str not in answer_text:
        with st.expander("Sources used"):
            # ensure docs is iterable
            candidate_docs = docs if isinstance(docs, (list, tuple)) else []

            # extract numbers (if numeric answer)
            try:
                ans_numbers = extract_numbers(answer_text)
            except Exception:
                ans_numbers = []

            # group candidates per source
            per_source_candidates = {}
            for d in candidate_docs:
                md = getattr(d, "metadata", {}) or {}
                src = md.get("source") or md.get("filename") or "source"
                page = md.get("page") or md.get("page_label") or ""
                label = src

                snippet_full = (d.page_content or "").strip()
                snippet_short = snippet_full[:2000] + (" …" if len(snippet_full) > 2000 else "")

                # scoring
                score = 0.0
                if ans_numbers:
                    try:
                        doc_numbers = extract_numbers(snippet_full)
                    except Exception:
                        doc_numbers = []
                    if any(n in doc_numbers for n in ans_numbers):
                        score = 1.0
                else:
                    try:
                        score = float(ngram_overlap(answer_text, snippet_full, n=5))
                    except Exception:
                        score = 0.0

                per_source_candidates.setdefault(label, []).append({
                    "page": page,
                    "score": score,
                    "snippet_short": snippet_short
                })

            # pick best page per source (deterministic tie-breaker: lower page number)
            selected_sources = []
            for src_label, entries in per_source_candidates.items():
                entries_sorted = sorted(
                    entries,
                    key=lambda e: (-e["score"], int(e["page"]) if str(e["page"]).isdigit() else 10**9)
                )
                best = entries_sorted[0]
                others = entries_sorted[1:] if len(entries_sorted) > 1 else []
                selected_sources.append((src_label, best, others))

            # display minimal output: filename + page + confidence + local file path
            FILE_URL = "/mnt/data/50d3186c-4849-4057-90b0-ccf66cd970ac.png"  # developer-provided local path

            if not selected_sources:
                st.write("No document pages were strongly matched to this answer.")
            else:
                for src_label, best, others in selected_sources:
                    conf = best.get("score", 0.0)
                    page_txt = f"(Page {best['page']})" if best.get("page") else "(Page unknown)"
                    badge = "✅" if conf >= 0.85 else ("⚠️" if conf >= 0.5 else "ℹ️")

                    # minimal, single line
                    st.write(f"{badge} **{src_label} {page_txt}** — Confidence: {conf:.2f}")

                    # show the local file path (your toolchain will convert to a URL)
                    st.write(f"🔗 File: `{FILE_URL}`")

               
