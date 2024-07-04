from dotenv import load_dotenv
import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
def get_context(product):
    prompt = f"{product} review"
    search = TavilySearchResults()
    op = search.invoke(prompt)
    content = [cur['content'] for cur in op]
    text = ""
    for t in content:
        text += t + "\n"
    return text 

def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    chunks = splitter.split_text(text)
    return chunks

def get_retriever(chunks):
    embedding = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_db = FAISS.from_texts(chunks,embedding)
    return vector_db.as_retriever()

def get_prompt():
    prompt = """
    You are a trained evaluator and problem solver. you will be given some
    review about a product and your role is to Let the user know the advantages ,
    disadvantages, special features and whether its value for money based on the 
    reviews provided to you

    Reviews : {context}
    """

    template = ChatPromptTemplate.from_messages(
        [
            ("system",prompt),
            ("human","{input}")
        ]
    )

    llm = ChatGoogleGenerativeAI(model = "gemini-1.5-pro")
    question_chain = create_stuff_documents_chain(llm,template)
    return question_chain

def process(input):
    text = get_context(input)
    chunks = get_chunks(text)
    retriever = get_retriever(chunks)
    return retriever

st.header("Product Rating")
input = st.text_input("Enter the product name", key = "input")
submit = st.button("submit")

if input or submit:
    if input is not None:
        retriever = process(input)
        template = get_prompt()
        rag = create_retrieval_chain(retriever,template)
        response = rag.invoke({'input': input})
        st.write(response['answer'])