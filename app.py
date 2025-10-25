import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
import os
from dotenv import load_dotenv
import base64

# --- Page Configuration ---
st.set_page_config(page_title="NBT Chatbot", page_icon="🤖", layout="wide")

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .st-emotion-cache-1y4p8pa {
        padding-top: 2rem;
    }
    /* You can add more specific chat message styling here if needed */
</style>
""", unsafe_allow_html=True)

#  Function to play audio
def autoplay_audio(file_path: str):
    """
    Embeds and autoplays an audio file in the Streamlit app.
    The function takes a file path, encodes the audio file to base64,
    and uses HTML to embed an invisible, auto-playing audio player.
    """
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true" style="display:none;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Audio file not found. Please ensure 'notification.mp3' is in the same directory.")
    except Exception as e:
        st.error(f"Error playing audio: {e}")

#  Environment Variable and API Key Loading 
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

#  Embedding Model 
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#  Session State Initialization 
if 'store' not in st.session_state:
    st.session_state.store = {}
if 'conversational_rag_chain' not in st.session_state:
    st.session_state.conversational_rag_chain = None



#  Sidebar for Controls 
with st.sidebar:
    st.header("Controls")
    
    
    session_id = st.text_input("Session ID", value="default_session")
    
    
    if st.button("Clear Chat History"):
        if session_id in st.session_state.store:
            st.session_state.store[session_id].clear()
            st.success("Chat history cleared!")
        else:
            st.info("No active chat history to clear.")
            
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, DOCX, PNG, JPG)",
        type=["pdf", "docx"], 
        accept_multiple_files=True
    )
#  Main App Interface 
st.title("🤖 NBT Chatbot")
st.write("Your intelligent assistant for PDF content. Upload documents in the sidebar to begin.")

#  API Key Check 
if not groq_api_key:
    st.warning("Groq API key not found. Please create a .env file and add GROQ_API_KEY='your_key'")
    st.stop()

#  PDF Processing and RAG Chain Creation 
if uploaded_files:
    with st.spinner("Processing documents... This may take a moment."):
        documents = []
        temp_dir = "temp_files"
        os.makedirs(temp_dir, exist_ok=True)
        
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = None
            file_name_lower = uploaded_file.name.lower()

            if file_name_lower.endswith(".pdf"):
                loader = PyPDFLoader(temp_path)
            elif file_name_lower.endswith(".docx"):
                loader = Docx2txtLoader(temp_path)
            
            else:
                st.warning(f"Skipping unsupported file: {uploaded_file.name}")

            if loader:
                try:
                    # .load() returns a list of Documents
                    documents.extend(loader.load()) 
                except Exception as e:
                    # Catch errors (e.g., Tesseract not found)
                    st.error(f"Error loading {uploaded_file.name}: {e}")
            
            os.remove(temp_path) 

        if not documents:
            st.error("No valid documents were processed. Please check your files or Tesseract installation.")
            st.stop()
            
            
        # --- This block now only appears ONCE ---
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

        # Contextualization prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Answering prompt
        system_prompt = (
            # --- MODIFIED LINE (Good Practice) ---
            "You are NBT Chatbot, an assistant specialized in answering questions about documents. " 
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        st.session_state.conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
    #  Only ONE success message at the end 
    st.success("Documents processed successfully! You can now ask questions.")


# Display chat messages from history
history = st.session_state.store.get(session_id, ChatMessageHistory()) 
for message in history.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# Accept user input
# Accept user input
if user_input := st.chat_input("Ask a question about your documents..."):
    with st.chat_message("human"):
        st.markdown(user_input)

    if st.session_state.conversational_rag_chain is None:
        st.warning("Please upload and process at least one PDF to begin.")
    else:
        # --- This is the BLOCK where 'response' is defined ---
        with st.spinner("NBT Chatbot is thinking..."):
            response = st.session_state.conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            
            with st.chat_message("ai"):
                st.markdown(response['answer'])
                # NOTE: The path to your audio file looks like an absolute path, 
                # which might cause issues if the code is run on a different machine.
                # Consider placing the audio file in the same directory or using a relative path.
                autoplay_audio(r"C:\Users\rautr\Desktop\Scalable AI Chatbot\new-notification-3-398649.mp3") 

                # --- START: The source document logic MUST be here ---
                source_documents = response.get('context', [])
                if source_documents:
                    
                    unique_sources = set()
                    for doc in source_documents:
                        source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                        page_num = doc.metadata.get('page', None) 
                        
                        if page_num is not None:
                            unique_sources.add(f"📄 **{source_name}** (Page: {page_num + 1})")
                        else:
                            # Fallback for DOCX or other file types without standard page numbers
                            unique_sources.add(f"📄 **{source_name}** (No Page Info)") 
                    
                    if unique_sources:
                        with st.expander("View Sources"):
                            st.markdown("---")
                            for source_info in sorted(list(unique_sources)):
                                st.markdown(source_info)
