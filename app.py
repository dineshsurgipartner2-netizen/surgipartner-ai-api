import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore

# --- 1. SETUP ---
# Your active API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyD7Fx0xMlQAJzbdyGdfPBBXlYdkCO509aE"

st.set_page_config(page_title="SurgiPartner AI", page_icon="🩺")
st.title("🩺 SurgiPartner Medical AI")
st.write("Official Assistant for LASIK Surgery Recovery - Hyderabad")

# --- 2. LOAD & PREPARE DATA ---
@st.cache_resource
def load_knowledge_base():
    # Verify the file exists in your surgipartner_senseai folder
    if not os.path.exists("lasik.txt"):
        st.error("❌ File 'lasik.txt' not found! Please ensure it is in the same folder as app.py")
        return None
        
    # Load and Split the text
    loader = TextLoader("lasik.txt") 
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    # Using the stable 2026 Embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Use InMemory to bypass Python 3.14 file-locking bugs
    vectorstore = InMemoryVectorStore.from_documents(splits, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 2})

retriever = load_knowledge_base()

# --- 3. AI LOGIC (STABLE CONNECTION) ---
# --- 3. AI LOGIC (2026 FREE TIER STABLE) ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # This is the new 2026 Free Tier standard
    temperature=0
)

system_prompt = (
    "You are the official medical chatbot for SurgiPartner Hyderabad. "
    "Use ONLY the provided context to answer. If unsure, say: "
    "'I can only provide information on verified topics. Please book an appointment with our Hyderabad clinic.'\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# --- 4. INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input field
if user_question := st.chat_input("How can I help with your LASIK recovery?"):
    st.chat_message("user").markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    with st.spinner("🔍 Searching SurgiPartner records..."):
        if retriever is None:
            st.error("Knowledge base could not be loaded.")
            st.stop()

        # Fetch the right info from lasik.txt
        found_docs = retriever.invoke(user_question)
        context_text = "\n".join([doc.page_content for doc in found_docs])
        
        # Ask the AI brain
        final_prompt = prompt.format_messages(context=context_text, input=user_question)
        response = llm.invoke(final_prompt)
        final_answer = response.content

    # Show the AI's answer
    with st.chat_message("assistant"):
        st.markdown(final_answer)
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
