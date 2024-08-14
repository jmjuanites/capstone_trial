import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.messages.utils import trim_messages

load_dotenv()

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
CHROMA_DATA_PATH = '/Users/PC/Documents/GitHub/eskwelabs_chatbot_streamlit/eskwelabs_capstone-main/embeddings_usecases_12_semantic'
COLLECTION_NAME = 'embeddings_usecases_12_semantic'

persistent_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection = persistent_client.get_collection(COLLECTION_NAME)

vectordb = Chroma(client=persistent_client, collection_name=COLLECTION_NAME, embedding_function=embedding_function)

llm = ChatOllama(
    model="llama3.1",
    temperature=0.5,
    num_predict=256,
    verbose=True
)

prompt = ChatPromptTemplate(
    messages=[
        MessagesPlaceholder(variable_name='chat_history'),
        ('system', 'You are a friendly assistant that answers questions on user inquiries.'),
        ('human', '{input}'),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

def resume_retriever_tool(url):
    loader = PyPDFLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(docs)

    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_function)
    retriever = vector_store.as_retriever()
    resume_retriever_tool = create_retriever_tool(
        retriever=retriever,
        name='resume_search',
        description='''Use this tool to parse the user's resume for details...'''
    )

    return resume_retriever_tool

def create_db_retriever_tools(vectordb):
    retriever_eskwelabs = vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "filter": {"use_case": {"$eq": "eskwelabs_faqs"}}}
    )

    eskwelabs_bootcamp_info_search_tool = create_retriever_tool(
        retriever=retriever_eskwelabs,
        name="eskwelabs_bootcamp_info_search",
        description='''Use this tool to retrieve comprehensive information about Eskwelabs...'''
    )

    retriever_bootcamp_vs_alternatives = vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "filter": {"use_case": {"$eq": "bootcamp_vs_selfstudy"}}}
    )

    bootcamp_vs_alternatives_search_tool = create_retriever_tool(
        retriever=retriever_bootcamp_vs_alternatives,
        name="bootcamp_vs_alternatives_search",
        description='''Use this tool to retrieve information about the pros and cons of bootcamps...'''
    )

    return eskwelabs_bootcamp_info_search_tool, bootcamp_vs_alternatives_search_tool

resume_tool = resume_retriever_tool('C:/Users/PC/Documents/GitHub/eskwelabs_chatbot_streamlit/Lasam_Resume.pdf')
eskwelabs_bootcamp_info_search_tool, bootcamp_vs_alternatives_search_tool = create_db_retriever_tools(vectordb)
tools = [resume_tool, eskwelabs_bootcamp_info_search_tool, bootcamp_vs_alternatives_search_tool]

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def process_chat(agent_executor, user_input, chat_history):
    response = agent_executor.invoke(
        {'input': user_input,
         'chat_history': chat_history
         },
    )
    return response['output']

# Streamlit App
st.title("Askwelabs")

# File Upload Section
st.subheader("Upload and Process Your Document Files")
docx_file = st.file_uploader("Upload File", type=['txt', 'docx', 'pdf'])

if docx_file:
    file_details = {
        "Filename": docx_file.name,
        "FileType": docx_file.type,
        "FileSize": docx_file.size
    }
    st.write(file_details)

    # Check File Type and Process Accordingly
    if docx_file.type == "application/pdf":
        with open("uploaded_file.pdf", "wb") as f:
            f.write(docx_file.getbuffer())
        resume_tool = resume_retriever_tool("uploaded_file.pdf")
        st.success("File content is ready to be used in the chatbot.")

# Chatbot Section
st.subheader("Eskwelabs Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Accept user input
if prompt := st.chat_input("Ask me anything!"):
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    # Process chat
    response = process_chat(agent_executor, prompt, st.session_state.chat_history)
    st.session_state.chat_history.append(AIMessage(content=response))

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        st.chat_message("user").markdown(message.content)
    elif isinstance(message, AIMessage):
        st.chat_message("assistant").markdown(message.content)
