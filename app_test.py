import streamlit as st
from pypdf import PdfReader
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableMap
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from PIL import Image
import os
from dotenv import load_dotenv


def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vector_database(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectordb


def get_rag_chain(hybrid_retriever):
    system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know based on the context but you can provide the answer if you know the answer without this context.No need to keep the answer precise"
    "\n\n"
    "{context}"
    )
    output = StrOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name='history'),
        ("human", "{input}"),
    ])
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    rag_chain = RunnableMap({
        "context": lambda x: hybrid_retriever.invoke(x["input"]),
        "input": lambda x: x["input"],
        'history' : lambda x :x['history']
    }) | prompt | model | output

    return rag_chain


def handle_userinput(user_question,history):
    response = st.session_state.rag.invoke({"input": user_question,'history':st.session_state.history})

    
    st.session_state.history.append(HumanMessage(content=user_question))
    st.session_state.history.append(AIMessage(content=response))
    #st.write(history)

    context_docs = st.session_state.retriever.invoke(user_question)
    evaluator_prompt = f"""
You are an evaluator for a Retrieval-Augmented Generation (RAG) system. Your task is to assess the quality of a response given the retrieved documents and the user's question.

Evaluate the following:

 1. Document Relevance
- Do the retrieved documents contain information that is relevant to the user's question?
- Rate relevance from 1 (not relevant) to 5 (highly relevant).

 2. Response Grounding
- Is the AI's response clearly supported by the retrieved documents?
- Rate grounding from 1 (not grounded at all) to 5 (fully grounded).
- If it includes external knowledge, is it clearly separated or labeled as such?

 3. Overall Response Quality
- Is the response correct, complete, clear, and useful to the user? Judge based on what the user asked and what was retrieved.
- Rate quality from 1 (poor) to 5 (excellent).

---

**User Question**:
{user_question}

**Retrieved Documents**:
{context_docs}

**AI Response**:
{response}

---

"""
    llm = genai.GenerativeModel('gemini-2.0-flash')  
    eval_result = llm.generate_content(evaluator_prompt).text

    st.session_state.chat_history.append({
        "sender": "user",
        "msg": user_question
    })
    st.session_state.chat_history.append({
        "sender": "ai",
        "msg": response,
        "context": context_docs,
        "eval": eval_result
    })
    return eval_result

def handle_without_pdf(user_question,history):
    llm = genai.GenerativeModel('gemini-2.0-flash')
    response = llm.generate_content(user_question)
    history.append(HumanMessage(content=user_question))
    history.append(AIMessage(content=response))
    st.session_state.chat_history.append(("user", user_question))
    st.session_state.chat_history.append(("ai", response.text))


def handle_image(query, image):
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content([query, image])
    st.session_state.chat_history.append(("user", query))
    st.session_state.chat_history.append(("ai", response.text))


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs/Images", page_icon=":books:")

    if "rag" not in st.session_state:
        st.session_state.rag = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with your PDFs/Images :books:")
    user_question = st.chat_input("Ask a question from these PDFs/Image")

    pdf_docs = st.session_state.get("pdf_docs", [])
    with st.sidebar:
        st.subheader("Your Documents/Images")

        pdf_docs = st.file_uploader("Upload your files here")
        if pdf_docs and st.button("Upload"):
            with st.spinner("Processing"):
                if pdf_docs.name.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):
                    st.session_state.upload_type = "image"
                    st.session_state.uploaded_image = Image.open(pdf_docs)
                    st.success("Image processed successfully!")

                elif pdf_docs.name.endswith('.pdf'):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectordb = get_vector_database(text_chunks)
                    retriever = vectordb.as_retriever(kwargs={"k": 5})
                    keyword_search = BM25Retriever.from_texts(text_chunks)
                    keyword_search.k = 5
                    hybrid_retriever = EnsembleRetriever(retrievers=[retriever, keyword_search], weights=[0.5, 0.5])
                    st.session_state.retriever = hybrid_retriever
                    st.session_state.rag = get_rag_chain(hybrid_retriever)
                    st.success("PDF processed successfully!")

    if "history" not in st.session_state:
        st.session_state.history=[
            HumanMessage(content='Hello'),
            AIMessage(content='Hi,how may I assist you?')
    ]
    if user_question:
        if pdf_docs and pdf_docs.name.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):
            st.image(st.session_state.uploaded_image, caption="Your Uploaded Image")
            handle_image(user_question, st.session_state.uploaded_image)
            
        elif pdf_docs and pdf_docs.name.endswith('.pdf'):
            answer=handle_userinput(user_question,st.session_state.history)
            
        else:
            handle_without_pdf(user_question,st.session_state.history)

    # Display chat history
    for entry in st.session_state.chat_history:
        with st.chat_message(entry["sender"]):
            st.markdown(entry["msg"])

        # Only AI responses have context + eval
        if entry["sender"] == "ai" and 'context' in entry:
            with st.expander("🔍 Retrieved Documents (Context)"):
                for i, doc in enumerate(entry["context"]):
                    st.markdown(f"**Doc {i+1}:** {doc}")
        
            with st.expander("🧪 Evaluation"):
                st.markdown(entry["eval"])


if __name__ == '__main__':
    main()
