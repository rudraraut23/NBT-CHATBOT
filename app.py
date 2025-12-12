import os
import time
import json
import hashlib
import traceback
import re
import uuid
import shutil
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Sequence
import base64
import pandas as pd

import streamlit as st
from dotenv import load_dotenv
import nest_asyncio

def play_notification_sound():
    try:
        audio_path = os.path.join(os.path.dirname(__file__), "notification.mp3")
        if not os.path.exists(audio_path):
            return

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        b64 = base64.b64encode(audio_bytes).decode("utf-8")
        audio_html = f"""
        <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception:
       
        pass


try:
    import fitz  
except Exception:
    fitz = None

try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory

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


UPLOADS_DIR = "uploaded_uploads"
TEMP_DIR = "tmp_parse"
HASH_STORE = "indexed_hashes.json"
FAISS_DIR = "faiss_index"
CHROMA_COLLECTION = "advanced_rag"

WORKERS = max(1, min(8, cpu_count() - 1))
CHUNK_SIZE_DEFAULT = 800
CHUNK_OVERLAP_DEFAULT = 200
EMBED_BATCH_SIZE = 128
MIN_TEXT_LEN = 30
RETRIEVE_K_DEFAULT = 5  

DEFAULT_MODEL_NAME = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.0

DEV_UPLOADED_FILE_URL = "/mnt/data/068227ba-2318-46e7-a76a-c0e73485ed39.png"

CLEAN_STARTUP = True
PDF_MAX_PAGES = None


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


def extract_pdf_pages(path: str, max_pages=None) -> List[Tuple[int, str]]:
    res = []
    if fitz is None:
        return res
    try:
        doc = fitz.open(path)
        total = len(doc)
        limit = total if max_pages is None else min(total, max_pages)
        for i in range(limit):
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


def parse_file_single(args):
    """Helper for parallel parsing. args = (path, orig_name, llama_api_key)"""
    path, orig_name, llama_api_key = args
    docs = []
    lower = orig_name.lower()

    if lower.endswith(".pdf") and fitz is not None:
        pages = extract_pdf_pages(path, max_pages=PDF_MAX_PAGES)
        for pnum, text in pages:
            if text and len(text.strip()) >= MIN_TEXT_LEN:
                meta = {"source": orig_name, "orig_name": orig_name, "page": int(pnum)}
                docs.append(Document(page_content=text, metadata=meta))
        return docs


    if lower.endswith((".png", ".jpg", ".jpeg")):
        txt = ocr_image_to_text(path)
        if txt and len(txt.strip()) >= MIN_TEXT_LEN:
            meta = {"source": orig_name, "orig_name": orig_name, "page": 1}
            docs.append(Document(page_content=txt, metadata=meta))
        return docs


    try:
        parser = LlamaParse(
            api_key=llama_api_key,
            result_type="markdown",
            verbose=False,
            language="en",
        )
        maybe = parser.load_data(path)
        if hasattr(maybe, "__iter__"):
            for idx, o in enumerate(maybe):
                text = getattr(o, "text", None) or getattr(o, "content", None) or ""
                if text and len(text.strip()) >= MIN_TEXT_LEN:
                    meta = getattr(o, "metadata", {}) or {}
                    page_lbl = meta.get("page_label", idx + 1)
                    meta_out = {
                        "source": orig_name,
                        "orig_name": orig_name,
                        "page": page_lbl,
                        **meta,
                    }
                    docs.append(Document(page_content=text, metadata=meta_out))
    except Exception:
        pass

    return docs


class SimpleEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        arr = self.model.encode(
            list(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=EMBED_BATCH_SIZE,
        )
        return [a.tolist() for a in arr]

    def embed_query(self, text: str) -> List[float]:
        arr = self.model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return arr[0].tolist()


def init_vectorstore_chroma(embeddings):
    if not CHROMA_AVAILABLE:
        raise RuntimeError("Chroma is not available")
    vs = Chroma(collection_name=CHROMA_COLLECTION, embedding_function=embeddings)
    return vs


def init_vectorstore_faiss_from_texts(texts, metas, embeddings):
    if not FAISS_AVAILABLE:
        raise RuntimeError("FAISS not available")
    return FAISS.from_texts(texts, embeddings, metadatas=metas)


def extract_numbers(s: str):
    return re.findall(r"\d+(?:\.\d+)?", s)


def ngram_overlap(a: str, b: str, n: int = 5) -> float:
    atoks = [t for t in re.findall(r"\w+", a.lower())]
    btoks = [t for t in re.findall(r"\w+", b.lower())]
    if len(atoks) < n:
        return 0.0
    a_ngrams = {" ".join(atoks[i : i + n]) for i in range(len(atoks) - n + 1)}
    b_text = " ".join(btoks)
    matches = sum(1 for ng in a_ngrams if ng in b_text)
    return matches / max(1, len(a_ngrams))


def answer_doc_overlap(answer: str, doc_text: str) -> float:
    """
    Rough lexical overlap between the final answer and a document snippet.
    Used to filter out retrieved chunks that clearly didn't contribute.
    Returns a value between 0 and 1.
    """
    if not answer or not doc_text:
        return 0.0
    a_tokens = set(re.findall(r"\w+", answer.lower()))
    d_tokens = set(re.findall(r"\w+", doc_text.lower()))
    if not a_tokens or not d_tokens:
        return 0.0
    return len(a_tokens & d_tokens) / len(a_tokens)


def get_docs_with_scores(vs_obj, query: str, k: int = RETRIEVE_K_DEFAULT,
                         oversample_factor: int = 4, max_per_source: int = 3):
    """
    Retrieve documents with scores, and ensure we don't only use one file.

    - Oversamples (k * oversample_factor) from the vectorstore.
    - Groups by source (orig_name).
    - Takes up to `max_per_source` chunks from each source.
    - Returns up to final K docs sorted by score.

    This makes it much more likely that multiple files contribute to the answer
    when they are relevant.
    """
    out = []


    try:
        big_k = max(k * oversample_factor, k)
        docs_and_scores = vs_obj.similarity_search_with_score(query, k=big_k)
        for d, s in docs_and_scores:
            out.append({"doc": d, "score": float(s)})
    except Exception:
        out = []


    if not out:
        try:
            docs = vs_obj.similarity_search(query, k=k * oversample_factor)
            for d in docs:
                out.append({"doc": d, "score": 0.0})
        except Exception:
            return []


    grouped = {}
    for item in out:
        d = item["doc"]
        md = getattr(d, "metadata", {}) or {}
        src = md.get("orig_name") or md.get("source") or "source"
        grouped.setdefault(src, []).append(item)


    diversified = []
    for src, items in grouped.items():
        items_sorted = sorted(items, key=lambda x: -float(x.get("score", 0.0)))
        diversified.extend(items_sorted[:max_per_source])


    diversified_sorted = sorted(
        diversified, key=lambda x: -float(x.get("score", 0.0))
    )
    return diversified_sorted[:k]



load_dotenv()
ensure_dirs()

st.set_page_config(page_title="NBT Advanced RAG", page_icon="🤖", layout="wide")
st.markdown(
    "<style>.main{background-color:#0b0d10;color:#fff}</style>",
    unsafe_allow_html=True,
)


if CLEAN_STARTUP and "startup_cleaned" not in st.session_state:
    for key in [
        "file_paths",
        "file_hashes",
        "processed_file_names",
        "vectorstore",
        "vectorstore_kind",
        "rag_chain",
        "chat_history",
    ]:
        if key in st.session_state:
            del st.session_state[key]

    try:
        for f in os.listdir(UPLOADS_DIR):
            try:
                os.remove(os.path.join(UPLOADS_DIR, f))
            except Exception:
                pass
    except Exception:
        pass

    safe_rmtree(FAISS_DIR)
    if os.path.exists(HASH_STORE):
        try:
            os.remove(HASH_STORE)
        except Exception:
            pass

    st.session_state.startup_cleaned = True


groq_api_key = os.getenv("GROQ_API_KEY")
llama_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

CHUNK_SIZE = CHUNK_SIZE_DEFAULT
CHUNK_OVERLAP = CHUNK_OVERLAP_DEFAULT
RETRIEVE_K = RETRIEVE_K_DEFAULT

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = "uploader_1"


def hard_reset_uploader():
    st.session_state.uploader_key = f"uploader_{time.time()}"

    try:
        for f in os.listdir(UPLOADS_DIR):
            try:
                os.remove(os.path.join(UPLOADS_DIR, f))
            except Exception:
                pass
    except Exception:
        pass

    if os.path.exists(HASH_STORE):
        try:
            os.remove(HASH_STORE)
        except Exception:
            pass

    safe_rmtree(FAISS_DIR)

    for k in [
        "processed_file_names",
        "vectorstore",
        "vectorstore_kind",
        "rag_chain",
        "file_hashes",
        "file_paths",
        "chat_history",
    ]:
        if k in st.session_state:
            del st.session_state[k]

    st.session_state.processed_file_names = []
    st.session_state.file_hashes = {}
    st.session_state.file_paths = {}
    st.rerun()


uploaded_files = st.file_uploader(
    "Upload Documents (PDF, DOCX, Images). You may upload multiple times; uploaded files are retained until you clear them.",
    accept_multiple_files=True,
    type=["pdf", "docx", "png", "jpg", "jpeg"],
    key=st.session_state.uploader_key,
)

if st.button("Clear uploaded files and indexing"):
    hard_reset_uploader()


if SentenceTransformer is None:
    st.error("Install sentence-transformers: pip install sentence-transformers")
    st.stop()

import torch

use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
if use_cuda:
    st.info("GPU available — embeddings will run on GPU.")
else:
    st.info("Embeddings on CPU.")

sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
embeddings_wrapper = SimpleEmbeddings(sbert_model)


if "processed_file_names" not in st.session_state:
    st.session_state.processed_file_names = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "file_hashes" not in st.session_state:
    st.session_state.file_hashes = load_indexed_hashes()
if "file_paths" not in st.session_state:
    st.session_state.file_paths = {}

_normalized_paths = {}
for key, val in list(st.session_state.file_paths.items()):
    try:
        if isinstance(val, dict):
            if "orig_name" not in val and "path" in val:
                val["orig_name"] = os.path.basename(val["path"]).split("_", 1)[-1]
            _normalized_paths[key] = val
        elif isinstance(val, str):
            path = val
            orig_name = os.path.basename(path).split("_", 1)[-1]
            try:
                with open(path, "rb") as rf:
                    file_hash = md5_bytes(rf.read())
            except Exception:
                file_hash = None
            _normalized_paths[key] = {
                "orig_name": orig_name,
                "path": path,
                "hash": file_hash,
            }
        else:
            continue
    except Exception:
        continue
st.session_state.file_paths = _normalized_paths


st.session_state.processed_file_names = sorted(
    {
        info.get("orig_name")
        for info in st.session_state.file_paths.values()
        if isinstance(info, dict) and info.get("orig_name")
    }
)

st.title("🤖 NBT Advanced RAG Chatbot")


llm = None
if groq_api_key:
    try:
        llm = ChatGroq(model_name=DEFAULT_MODEL_NAME, temperature=LLM_TEMPERATURE)
    except Exception:
        st.warning(
            "Could not initialize Groq LLM; continuing with indexing-only features."
        )
else:
    st.warning("GROQ_API_KEY not found in environment. Chat will be disabled.")


new_files_to_process = []
if uploaded_files:
    for uf in uploaded_files:
        content = uf.getvalue()
        h = md5_bytes(content)

      
        already = False
        for info in st.session_state.file_paths.values():
            if info.get("hash") == h:
                already = True
                break
        if already:
            continue

        unique_prefix = uuid.uuid4().hex[:8]
        unique_on_disk_name = f"{unique_prefix}_{uf.name}"
        dest = os.path.join(UPLOADS_DIR, unique_on_disk_name)
        try:
            with open(dest, "wb") as wf:
                wf.write(content)
        except Exception:
            st.error(
                f"Cannot write uploaded file to disk: {uf.name}. Check server permission."
            )
            continue

        st.session_state.file_paths[unique_on_disk_name] = {
            "orig_name": uf.name,
            "path": dest,
            "hash": h,
        }
        new_files_to_process.append((uf.name, dest, h, unique_on_disk_name))


for fname in os.listdir(UPLOADS_DIR):
    if fname not in st.session_state.file_paths:
        path = os.path.join(UPLOADS_DIR, fname)
        try:
            with open(path, "rb") as rf:
                content = rf.read()
            h = md5_bytes(content)
            st.session_state.file_paths[fname] = {
                "orig_name": fname.split("_", 1)[-1],
                "path": path,
                "hash": h,
            }
        except Exception:
            pass


if new_files_to_process or st.session_state.vectorstore is None:
    if new_files_to_process:
        st.info("New uploads detected — rebuilding index to include latest files.")
    else:
        st.info("No vectorstore found — building index from available files.")

    st.session_state.vectorstore = None
    try:
        safe_rmtree(FAISS_DIR)
    except Exception:
        pass
    if os.path.exists(HASH_STORE):
        try:
            os.remove(HASH_STORE)
        except Exception:
            pass

    all_parent_docs = []
    parse_args = []
    for key, info in st.session_state.file_paths.items():
        parse_args.append((info["path"], info["orig_name"], llama_api_key))

    parse_start = time.time()
    errors = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(parse_file_single, a): a for a in parse_args}
        progress = st.progress(0)
        total = len(futures)
        done = 0
        for fut in as_completed(futures):
            done += 1
            progress.progress(min(100, int(done / max(1, total) * 100)))
            try:
                docs = fut.result()
                all_parent_docs.extend(docs)
            except Exception:
                errors.append(traceback.format_exc())
    parse_time = time.time() - parse_start

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(CHUNK_SIZE), chunk_overlap=int(CHUNK_OVERLAP)
    )
    all_texts = []
    all_metas = []
    for doc in all_parent_docs:
        chunks = splitter.split_documents([doc])
        for c in chunks:
            text = c.page_content or ""
            if len(text.strip()) < MIN_TEXT_LEN:
                continue
            meta = c.metadata or {}
            src = (
                meta.get("orig_name")
                or meta.get("source")
                or doc.metadata.get("orig_name")
                or doc.metadata.get("source")
            )
            page = meta.get("page") or doc.metadata.get("page") or ""

            try:
                page_num = int(page)
            except Exception:
                page_num = page
            chunk_id = (
                f"{src}_p{page_num}_"
                f"{hashlib.md5(text.encode()).hexdigest()[:8]}"
            )
            all_texts.append(text)
            all_metas.append(
                {"source": src, "orig_name": src, "page": page_num, "chunk_id": chunk_id}
            )

    def _create_vs_from_texts(texts, metas):
        try:
            if CHROMA_AVAILABLE:
                vs = init_vectorstore_chroma(embeddings_wrapper)
                try:
                    vs.add_texts(texts, metadatas=metas)
                except Exception:
                    vs = vs.from_texts(texts, embeddings_wrapper, metadatas=metas)
                return "chroma", vs
            else:
                raise RuntimeError("Chroma not available")
        except Exception:
            vs = init_vectorstore_faiss_from_texts(texts, metas, embeddings_wrapper)
            try:
                vs.save_local(FAISS_DIR)
            except Exception:
                pass
            return "faiss", vs

    try:
        kind, vectorstore = _create_vs_from_texts(all_texts, all_metas)
        st.session_state.vectorstore = vectorstore
        st.session_state.vectorstore_kind = kind
        st.success(
            f"Index built ({kind}) — {len(all_texts)} chunks from "
            f"{len(st.session_state.file_paths)} files (parse time: {parse_time:.1f}s)."
        )
    except Exception:
        st.error("Failed to build index; check parse_errors.log")
        with open("parse_errors.log", "a") as lf:
            lf.write(traceback.format_exc())


    for orig_name, path, h, unique_key in new_files_to_process:
        st.session_state.file_hashes[unique_key] = h
    save_indexed_hashes(st.session_state.file_hashes)

st.session_state.processed_file_names = sorted(
    [info["orig_name"] for info in st.session_state.file_paths.values()]
)

with st.expander("Indexed files (persistent) — click to view"):
    st.write(
        {
            k: {
                "orig_name": v["orig_name"],
                "hash": v["hash"],
                "path": v["path"],
            }
            for k, v in st.session_state.file_paths.items()
        }
    )
    st.write(
        {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "min_text_len": MIN_TEXT_LEN,
        }
    )


if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()
for msg in st.session_state.chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

answer_text = None
docs_sorted = []

user_input = st.chat_input("Ask about your documents (legal/CA friendly)...")
if user_input:
    st.chat_message("human").write(user_input)

    identity_triggers = [
        "who are you",
        "about yourself",
        "what are you",
        "your identity",
        "what do you do",
    ]
    is_identity = any(t in user_input.lower() for t in identity_triggers)
    handled = False

    if is_identity:
        ai_resp = (
            "I am NBT Advanced RAG — I answer questions using your uploaded "
            "documents (CA / legal friendly)."
        )
        st.chat_message("ai").write(ai_resp)
        play_notification_sound()
        try:
            st.session_state.chat_history.add_user_message(user_input)
            st.session_state.chat_history.add_ai_message(ai_resp)
        except Exception:
            pass
        handled = True

    if not handled and st.session_state.vectorstore is None:
        st.warning("Please upload and index documents first.")
        handled = True

    if not handled and llm is None:
        st.error("LLM is not initialized. Please set GROQ_API_KEY in your environment.")
        handled = True

    if not handled:
        vs = st.session_state.vectorstore

       
        docs_with_scores = get_docs_with_scores(
            vs,
            user_input,
            k=int(RETRIEVE_K),
            oversample_factor=4,
            max_per_source=3,
        )
        docs_sorted = sorted(
            docs_with_scores, key=lambda x: -float(x.get("score", 0.0))
        )

        context_pieces = []
        for item in docs_sorted:
            d = item["doc"]
            md = getattr(d, "metadata", {}) or {}
            src = md.get("orig_name") or md.get("source") or "source"
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
            "\"I've checked the documents, but I can't find a clear answer to that specific question.\" "
            "You may provide brief clarifications or next steps, but do NOT hallucinate facts not supported by the documents "
            "unless explicitly asked for background.\n\n"
            "Context:\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        chain = qa_prompt | llm


        try:
            with st.spinner("Thinking (using retrieved context)..."):
                response = chain.invoke(
                    {
                        "input": user_input,
                        "context": full_context,
                        "chat_history": st.session_state.chat_history.messages,
                    }
                )
        except Exception:
            st.error("RAG chain error; see parse_errors.log")
            with open("parse_errors.log", "a") as lf:
                lf.write(traceback.format_exc())
            response = None

        if response is None:
            answer_text = (
                "Based on the documents provided, I cannot answer this question."
            )
        else:
            answer_text = getattr(response, "content", None) or str(response)


        st.chat_message("ai").write(answer_text)
        play_notification_sound()
        try:
            st.session_state.chat_history.add_user_message(user_input)
            st.session_state.chat_history.add_ai_message(answer_text)
        except Exception:
            pass


if answer_text is not None:
    refusal_str = (
        "I've checked the documents, but I can't find a clear answer to that specific question."
    )
    if refusal_str not in answer_text:
        with st.expander("Sources used"):
            local_docs_sorted = docs_sorted or []

            if not local_docs_sorted:
                st.write("No document pages were strongly matched to this answer.")
            else:
                OVERLAP_THRESHOLD = 0.45

                filtered = []
                for item in local_docs_sorted:
                    d = item.get("doc")
                    snippet = (d.page_content or "").strip()
                    ov = answer_doc_overlap(answer_text, snippet[:1000])
                    if ov >= OVERLAP_THRESHOLD:
                        new_item = dict(item)
                        new_item["overlap"] = ov
                        filtered.append(new_item)

                if not filtered:
                    st.write(
                        "Retrieved pages had very low overlap with the final answer; no reliable sources to show."
                    )
                else:
                    grouped = {}
                    for item in filtered:
                        d = item.get("doc")
                        md = getattr(d, "metadata", {}) or {}
                        src = md.get("orig_name") or md.get("source") or "source"
                        page = md.get("page") or md.get("page_label") or ""
                        key = (src, page)
                        grouped.setdefault(key, []).append(item)

                    display_rows = []
                    for (src, page), arr in grouped.items():
                        best_score = max(float(a.get("score", 0.0)) for a in arr)
                        best_overlap = max(float(a.get("overlap", 0.0)) for a in arr)
                        combined = best_score * 0.3 + best_overlap * 0.7
                        display_rows.append(
                            ((src, page), combined, best_score, best_overlap, arr)
                        )

                    display_rows.sort(key=lambda x: -x[1])

                    for (src, page), combined, best_score, best_overlap, arr in display_rows:
                        page_txt = (
                            f"(Page {page})"
                            if page not in (None, "", "unknown")
                            else "(Page unknown)"
                        )
                        badge = (
                            "✅"
                            if combined >= 0.7
                            else ("⚠️" if combined >= 0.4 else "ℹ️")
                        )

                        st.write(
                            f"{badge} **{src} {page_txt}** — "
                            f"Vec score: {best_score:.4f} · "
                            f"Overlap with answer: {best_overlap:.2f}"
                        )

                        file_url = None
                        for ondisk, info in st.session_state.file_paths.items():
                            try:
                                if (
                                    info.get("orig_name") == src
                                    or ondisk.endswith(src)
                                    or info.get("path", "").endswith(src)
                                ):
                                    file_url = info.get("path")
                                    break
                            except Exception:
                                continue
