import os
import shutil
import streamlit as st
import time
import hashlib
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParser
from langchain_community.document_loaders import YoutubeAudioLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from decouple import config
import sys

sys.stdout.reconfigure(encoding="utf-8")

# Streamlit app configuration
st.set_page_config(page_title="ChatTube", layout="centered")
st.title("ChatTube - Chat with any YouTube video")

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")

# Directory for storing audio & vector data
SAVE_DIR = "./YouTube"
VECTOR_STORE_BASE_DIR = "vector_store"

# Initialize session state variables
if "current_video_id" not in st.session_state:
    st.session_state["current_video_id"] = None

if "qa_chain" not in st.session_state:
    st.session_state["qa_chain"] = None

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

# Function to generate a unique ID for each video URL
def get_video_id(url):
    return hashlib.md5(url.encode()).hexdigest()

# Function to transcribe YouTube video
def transcribe_video(url):
    loader = GenericLoader(YoutubeAudioLoader([url], SAVE_DIR), OpenAIWhisperParser())
    docs = loader.load()
    combined_docs = [doc.page_content for doc in docs]
    return " ".join(combined_docs)

# Function to split text into chunks
def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    return text_splitter.split_text(text)

# Function to build vector store for a specific video
def build_vector_store(video_id, splits):
    vector_store_path = os.path.join(VECTOR_STORE_BASE_DIR, video_id)
    os.makedirs(vector_store_path, exist_ok=True)  # Ensure directory exists

    embedding_function = OpenAIEmbeddings()
    vectordb = Chroma.from_texts(
        splits, embedding_function, persist_directory=vector_store_path, collection_name="youtube_video"
    )
    vectordb.persist()
    return vectordb

# Function to load existing vector store
def load_vector_store(video_id):
    vector_store_path = os.path.join(VECTOR_STORE_BASE_DIR, video_id)
    if os.path.exists(vector_store_path):
        return Chroma(
            persist_directory=vector_store_path,
            collection_name="youtube_video",
            embedding_function=OpenAIEmbeddings(),
        )
    return None  # Return None if vector store doesn't exist

# Function to set up QA Chain
def setup_qa_chain(vectordb):
    prompt = PromptTemplate(
        template="""Given the context about a video, answer the user in a friendly and precise manner.

        Context: {context}

        Human: {question}

        AI:""",
        input_variables=["context", "question"],
    )
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

# User inputs video URL
video_url = st.text_input("Enter the YouTube video URL:")

if video_url:
    video_id = get_video_id(video_url)
    vector_store_path = os.path.join(VECTOR_STORE_BASE_DIR, video_id)

    # Check if the vector store for this video already exists
    vectordb = load_vector_store(video_id)

    if vectordb:
        st.write("Using existing embeddings for this video.")
    else:
        st.write("Processing new video and creating embeddings...")

        # Transcribe and process the video
        transcribed_text = transcribe_video(video_url)
        splits = split_text(transcribed_text)
        vectordb = build_vector_store(video_id, splits)

    # Update session state with the current video ID and QA chain
    st.session_state["current_video_id"] = video_id
    st.session_state["qa_chain"] = setup_qa_chain(vectordb)

    # **Chat UI**
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # **User input for chat**
    user_prompt = st.chat_input("Ask something about the video:")
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)

        # **Generate AI Response**
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state["qa_chain"]({"query": user_prompt})
                st.write(response["result"])

        st.session_state.messages.append({"role": "assistant", "content": response["result"]})
