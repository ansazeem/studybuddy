# main.py
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import google.generativeai as genai

from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# List all available models
models = genai.list_models()

# Print model names
print("Available models:")
for model in models:
    print(model.name)

pdf_path = "data/script.pdf"
loader = PyPDFLoader(pdf_path)

# This gives you a list of documents (one per page)
pages = loader.load()
print(f"Loaded {len(pages)} pages.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # Max characters per chunk
    chunk_overlap=50,      # Allow some overlap for context
)
chunks = text_splitter.split_documents(pages)
print(f"Split into {len(chunks)} chunks.")

# Create embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


# Create FAISS vector store from our chunks
vectorstore = FAISS.from_documents(chunks, embedding_model)

# Optional: Save it for later use
vectorstore.save_local("db/vectorstore")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",  # or "gemini-1.5-pro-latest" if available in your region
    temperature=0.3
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),  # This pulls top k matching chunks
    return_source_documents=True  # Optional: show which chunks were used
)

query = "what did i provide to"
response = qa_chain(query)

print("\nAnswer:")
print(response["result"])

# # Optional: Show source chunks
# print("\nSources:")
# for doc in response["source_documents"]:
#     print(doc.page_content[:200])  # Preview of source content
