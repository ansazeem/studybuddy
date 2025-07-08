# # main.py
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_huggingface import HuggingFaceEndpointEmbeddings
# from langchain.prompts import PromptTemplate
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
# from langchain_milvus import Milvus
# from langchain.prompts import PromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.schema.runnable import RunnableMap, RunnableLambda
# import google.generativeai as genai
# import csv

# from dotenv import load_dotenv
# import os

# from pymilvus import connections
# print("Connecting to Milvus...")
# connections.connect(host="localhost", port="19530")


# # load_dotenv()
# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# # # List all available models
# # models = genai.list_models()

# # # Print model names
# # print("Available models:")
# # for model in models:
# #     print(model.name)

# pdf_path = "data/coding_book.pdf"
# loader = PyPDFLoader(pdf_path)

# # This gives you a list of documents (one per page)
# pages = loader.load()
# print(f"Loaded {len(pages)} pages.")

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=512,        # Max characters per chunk
#     chunk_overlap=64    # Allow some overlap for context
# )

# chunks = text_splitter.split_documents(pages)
# print(f"Split into {len(chunks)} chunks.")
# # # 📊 Visualize chunks
# # for i, chunk in enumerate(chunks):
# #     print(f"\n--- Chunk {i+1} ---")
# #     print(f"Length: {len(chunk.page_content)} chars")
# #     print(chunk.page_content[:300])  # Show first 300 chars

# #     import csv

# with open("chunks_preview.csv", "w", newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(["Chunk Number", "Length", "Content (First 300 Chars)"])
#     for i, chunk in enumerate(chunks):
#         writer.writerow([i + 1, len(chunk.page_content), chunk.page_content[:300]])


# # 3. Embed with Qwen3
# embedding_model = HuggingFaceEndpointEmbeddings(
#     repo_id="Qwen/Qwen3-Embedding-0.6B",  # The embedding model you want
#     task="feature-extraction",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
# )

# # 4. Store in Milvus
# vectorstore = Milvus.from_documents(
#     documents=chunks,
#     embedding=embedding_model,
#     connection_args={
#         "host": "localhost",
#         "port": "19530"
#     },
#     collection_name="studybuddy_docs"
# )
# retriever = vectorstore.as_retriever()

# # 5. Build prompt + chain
# prompt = PromptTemplate.from_template("""You are a helpful study assistant.

# Use the context to answer the question clearly.

# Context:
# {context}

# Question:
# {question}
# """)

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)

# rag_chain = (
#     RunnableMap({"context": retriever | RunnableLambda(format_docs), "question": lambda x: x["question"]})
#     | prompt
#     | llm
# )

# # 6. Ask a question
# question = {"question": "What is classical conditioning?"}
# response = rag_chain.invoke(question)

# print("\n📚 Answer:")
# print(response.content)

from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
# main.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_milvus import Milvus
import google.generativeai as genai
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from yaspin import yaspin

load_dotenv()
import os

pdf_path = "data/LangChain.pdf"
loader = PyPDFLoader(pdf_path)

# This gives you a list of documents (one per page)
pages = loader.load()
print(f"Loaded {len(pages)} pages.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,        # Max characters per chunk
    chunk_overlap=64    # Allow some overlap for context
)
chunks = text_splitter.split_documents(pages)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


# Create FAISS vector store from our chunks
vectorstore = FAISS.from_documents(chunks, embedding_model)

# Optional: Save it for later use
vectorstore.save_local("db/vectorstore")
retriever=vectorstore.as_retriever()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",  # or "gemini-1.5-pro-latest" if available in your region
    temperature=0.3
)


prompt = ChatPromptTemplate.from_template("""
You are an AI tutor that is skilled at explaining information. Use the provided context below to answer the given question. 
Do not simply state the information provided in the context but also explain it. Use simple and easy to understand vocabulary 
while maintaining clarity in your responses. Do not include things like "Based on the context" or "According to the information 
provided". The end user should not be aware of the system inxstructions given to you.
If the answer cannot be found in the context, say "Based on the provided context, I do not have sufficient information"."

Context: {context}

Question: {question}
""")

# 2. Create a retrieval chain
retriever_chain = (
    RunnableMap({"context": lambda x: retriever.get_relevant_documents(x["question"]), "question": lambda x: x["question"]})
    | prompt
    | llm
    | StrOutputParser()
)

# 3. Run the chain
retriever_chain = (
    RunnableMap({
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"]
    })
    | prompt
    | llm
    | StrOutputParser()
)


question_input = {"question": "What is LangChain?"}

with yaspin(text="🤖 Thinking...", color="green") as spinner:
    response = retriever_chain.invoke(question_input)
    spinner.ok("✅") 

print("\nAnswer:")
print(response)
