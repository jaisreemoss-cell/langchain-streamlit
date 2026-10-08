import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from a .env file if present
load_dotenv()

# -----------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Multi-PDF RAG Assistant", page_icon="📚", layout="wide"
)

st.title("📚 Multi-PDF RAG Assistant")
st.markdown(
    "Upload multiple PDF documents, index them into a shared vector store, and ask questions with exact source citations (file name and page number)."
)

# -----------------------------------------------------------------
# Sidebar: Configuration & Document Upload
# -----------------------------------------------------------------
with st.sidebar:
  st.header("1. Configuration")

  # Pre-fill with environment variable loaded via python-dotenv (or system env)
  default_api_key = os.environ.get("OPENAI_API_KEY", "")

  api_key_input = st.text_input(
      "OpenAI API Key",
      type="password",
      value=default_api_key,
      help=(
          "Loaded automatically from your .env file or environment variable if"
          " set."
      ),
  )

  if api_key_input:
    os.environ["OPENAI_API_KEY"] = api_key_input

  st.header("2. Upload Documents")
  uploaded_files = st.file_uploader(
      "Upload PDF files", type=["pdf"], accept_multiple_files=True
  )

  process_button = st.button("Process Documents", type="primary")

# -----------------------------------------------------------------
# Initialize Session State
# -----------------------------------------------------------------
if "vector_store" not in st.session_state:
  st.session_state.vector_store = None
if "messages" not in st.session_state:
  st.session_state.messages = []

# -----------------------------------------------------------------
# Processing Pipeline
# -----------------------------------------------------------------
if process_button:
  if not os.environ.get("OPENAI_API_KEY"):
    st.error(
        "Please provide a valid OpenAI API key in your .env file, sidebar, or"
        " environment variables."
    )
  elif not uploaded_files:
    st.error("Please upload at least one PDF file.")
  else:
    with st.spinner(
        "Processing PDFs: loading, chunking, and embedding into FAISS..."
    ):
      all_docs = []

      # Temporary directory to safely handle uploaded files
      with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
          temp_file_path = os.path.join(temp_dir, uploaded_file.name)
          with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

          # Load PDF using PyPDFLoader (retains page metadata)
          loader = PyPDFLoader(temp_file_path)
          docs = loader.load()

          # Standardize source metadata to just the file name
          for doc in docs:
            doc.metadata["source"] = uploaded_file.name

          all_docs.extend(docs)

        # Chunk the documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(all_docs)

        # Embed and store in FAISS
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.from_documents(chunks, embeddings)

        # Save to session state
        st.session_state.vector_store = vector_store
        st.success(
            f"Successfully indexed {len(chunks)} chunks from"
            f" {len(uploaded_files)} PDF(s) into the vector store!"
        )

# -----------------------------------------------------------------
# Chat Interface & RAG Execution
# -----------------------------------------------------------------
if st.session_state.vector_store is not None:
  st.divider()
  st.subheader("Ask Questions Across Your Documents")

  # Display chat history
  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

  # Chat input
  if user_query := st.chat_input(
      "Ask a question about your uploaded PDFs..."
  ):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
      st.markdown(user_query)

    with st.chat_message("assistant"):
      with st.spinner("Searching documents and generating answer..."):
        # Setup retriever and LLM
        retriever = st.session_state.vector_store.as_retriever(
            search_kwargs={"k": 4}
        )
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


        # Format retrieved documents for prompt and extraction
        def format_docs(docs):
          formatted = []
          for d in docs:
            source_file = d.metadata.get("source", "Unknown file")
            page_num = (
                d.metadata.get("page", 0) + 1
            )  # PyPDF page numbers are 0-indexed
            formatted.append(
                f"Source File: {source_file} | Page: {page_num}\nContent:"
                f" {d.page_content}"
            )
          return "\n\n---\n\n".join(formatted)


        retrieved_docs = retriever.invoke(user_query)
        context_str = format_docs(retrieved_docs)

        prompt = ChatPromptTemplate.from_template(
            """Answer the user's question based strictly on the provided context below. 
If you do not know the answer based on the context, state that you cannot find the answer in the documents.
Always cite your sources by mentioning the specific file name and page number for the claims you make.

Context:
{context}

Question: {question}
"""
        )

        chain = (
            {"context": lambda x: context_str, "question": RunnablePassthrough()}
            | prompt
            | llm
        )

        response = chain.invoke(user_query)
        answer_text = response.content

        # Append source badges / card summary for transparency
        sources_md = "\n\n**Sources Consulted:**"
        seen_sources = set()
        for d in retrieved_docs:
          s_file = d.metadata.get("source", "Unknown")
          p_num = d.metadata.get("page", 0) + 1
          identifier = (s_file, p_num)
          if identifier not in seen_sources:
            seen_sources.add(identifier)
            sources_md += f"\n- **{s_file}** (Page {p_num})"

        full_response = answer_text + sources_md
        st.markdown(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
else:
  st.info(
      "👈 Please upload your PDF files in the sidebar and click **Process"
      " Documents** to begin."
  )