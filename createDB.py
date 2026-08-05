# load pdf
#split pdf into chunks
#create embeddings for each chunk
#store embeddings in vector database

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma



load_dotenv()


data = PyPDFLoader(r"C:\courseinfo\document loaders\MachineLearning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)

vectorstore = Chroma.from_documents(
    documents= chunks,
    embedding=embedding_model,
    persist_directory="chroma"
)