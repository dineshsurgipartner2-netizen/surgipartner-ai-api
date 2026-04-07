import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from dotenv import load_dotenv

# 1. SETUP & SECURITY
load_dotenv() 

app = FastAPI(
    title="SurgiPartner AI API",
    description="Backend API for LASIK Surgery Recovery Assistant - Hyderabad"
)

# 2. DEFINE THE DATA FORMAT FOR THE APP TEAM
class ChatRequest(BaseModel):
    user_question: str

retriever = None
llm = None
prompt_template = None

# 3. LOAD KNOWLEDGE BASE ON STARTUP
@app.on_event("startup")
def startup_event():
    global retriever, llm, prompt_template
    
    if not os.path.exists("lasik.txt"):
        print("❌ Error: 'lasik.txt' not found!")
        return

    loader = TextLoader("lasik.txt") 
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = InMemoryVectorStore.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    system_prompt = (
        "You are the official medical chatbot for SurgiPartner Hyderabad. "
        "Use ONLY the provided context to answer. If unsure, say: "
        "'I can only provide information on verified topics. Please book an appointment with our Hyderabad clinic.'\n\n"
        "Context: {context}"
    )
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    print("✅ SurgiPartner AI successfully loaded and ready.")

# 4. THE API ENDPOINT
@app.post("/ask")
async def ask_medical_question(request: ChatRequest):
    if retriever is None:
        raise HTTPException(status_code=500, detail="Knowledge base not loaded.")

    try:
        found_docs = retriever.invoke(request.user_question)
        context_text = "\n".join([doc.page_content for doc in found_docs])
        
        final_prompt = prompt_template.format_messages(context=context_text, input=request.user_question)
        response = llm.invoke(final_prompt)
        
        return {"answer": response.content}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
