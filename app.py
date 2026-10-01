import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="centered"
)


# =========================================================
# INTERNAL CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #e3f2fd, #e8f5e9);
}

.block-container {
    max-width: 850px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #1565c0;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555555;
    margin-bottom: 25px;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    margin-bottom: 20px;
    box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.08);
}

.answer-box {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border-left: 5px solid #43a047;
    box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.08);
    line-height: 1.7;
}

[data-testid="stFileUploader"] {
    background: white;
    padding: 15px;
    border-radius: 15px;
    border: 2px dashed #90caf9;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: none;
    background: #1565c0;
    color: white;
    font-weight: bold;
    padding: 10px;
}

.stButton > button:hover {
    background: #0d47a1;
    color: white;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">📄 Chat with Your PDF</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about your PDF using RAG + Ollama'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# =========================================================
# CREATE CHROMADB
# =========================================================

@st.cache_resource
def create_chroma_database():

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="documents"
    )

    return collection


collection = create_chroma_database()


# =========================================================
# PDF UPLOAD
# =========================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("📂 Upload your PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # READ PDF
    # -----------------------------------------------------

    try:

        reader = PdfReader(
            uploaded_file
        )

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    except Exception as error:

        st.error(
            f"Unable to read the PDF: {error}"
        )

        st.stop()


    # -----------------------------------------------------
    # CHECK PDF TEXT
    # -----------------------------------------------------

    if not text.strip():

        st.error(
            "No text could be extracted from this PDF."
        )

        st.warning(
            "This may be a scanned or image-based PDF."
        )

        st.stop()


    st.success(
        "✅ PDF uploaded successfully!"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Pages",
            len(reader.pages)
        )

    with col2:

        st.metric(
            "Characters",
            len(text)
        )


    # =====================================================
    # CREATE CHUNKS
    # =====================================================

    chunk_size = 500

    chunks = []

    for start in range(
        0,
        len(text),
        chunk_size
    ):

        chunk = text[
            start:start + chunk_size
        ]

        if chunk.strip():

            chunks.append(chunk)


    st.info(
        f"🧩 Created {len(chunks)} text chunks."
    )


    # =====================================================
    # CREATE EMBEDDINGS
    # =====================================================

    with st.spinner(
        "Creating document embeddings..."
    ):

        embeddings = embedding_model.encode(
            chunks
        )


    st.success(
        "✅ Embeddings created successfully!"
    )


    # =====================================================
    # STORE IN CHROMADB
    # =====================================================

    try:

        # Delete previous documents

        old_data = collection.get()

        if old_data["ids"]:

            collection.delete(
                ids=old_data["ids"]
            )


        # Create unique IDs

        ids = [
            f"chunk_{i}"
            for i in range(len(chunks))
        ]


        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist()
        )


        st.success(
            "✅ PDF stored successfully in ChromaDB!"
        )

    except Exception as error:

        st.error(
            f"ChromaDB error: {error}"
        )

        st.stop()


    # =====================================================
    # QUESTION
    # =====================================================

    st.divider()

    st.subheader("💬 Ask a Question")

    question = st.text_input(
        "Ask something about your PDF",
        placeholder="Example: What is this document about?"
    )


    # =====================================================
    # RAG SEARCH
    # =====================================================

    if question.strip():

        with st.spinner(
            "🔍 Searching the document..."
        ):

            question_embedding = (
                embedding_model.encode(
                    [question]
                )
            )


            # Retrieve only top 5 relevant chunks

            results = collection.query(

                query_embeddings=
                question_embedding.tolist(),

                n_results=min(
                    5,
                    len(chunks)
                )
            )


        relevant_chunks = results[
            "documents"
        ][0]


        context = "\n\n".join(
            relevant_chunks
        )


        # =================================================
        # SHOW RETRIEVED CONTEXT
        # =================================================

        with st.expander(
            "🔎 View Retrieved Context"
        ):

            st.write(context)


        # =================================================
        # PROMPT
        # =================================================

        prompt = f"""
You are a helpful PDF document assistant.

Answer the user's question using only the
information provided in the document context.

Do not make up information.

If the answer is not available in the
document context, say:

"I could not find the answer in the document."

Document Context:
{context}

User Question:
{question}

Answer:
"""


        # =================================================
        # OLLAMA
        # =================================================

        with st.spinner(
            "🤖 Ollama is generating the answer..."
        ):

            try:

                response = ollama.chat(

                    model="llama3.2",

                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                answer = response[
                    "message"
                ][
                    "content"
                ]


                # =========================================
                # DISPLAY ANSWER
                # =========================================

                st.subheader(
                    "🤖 Answer"
                )

                st.markdown(
                    f"""
                    <div class="answer-box">
                        {answer}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            except Exception:

                st.error(
                    "Ollama is not running or the "
                    "llama3.2 model is not available."
                )

                st.info(
                    "Open Ollama and make sure "
                    "llama3.2 is installed."
                )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Built with Python • Streamlit • RAG • "
    "Sentence Transformers • ChromaDB • Ollama"
)
