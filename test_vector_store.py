from langchain_community.vectorstores import LanceDB
import lancedb
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatAnthropic
from langchain_openai import ChatOpenAI

embedding = OpenAIEmbeddings()

loader = PyPDFLoader("PA00TBCT.pdf")
pages = loader.load_and_split()

# Initialize the BM25 retriever
bm25_retriever = BM25Retriever.from_documents(pages)
bm25_retriever.k =  2  # Retrieve top 2 results

db = lancedb.connect('lancedb')
# Before creating the new table
if "pandas_docs" in db.table_names():
    db.drop_table("pandas_docs")
# Create table through the database connection instead of directly using the table
docsearch = LanceDB.from_documents(pages, embedding, connection=db, table_name="pandas_docs")
retriever_lancedb = docsearch.as_retriever(search_kwargs={"k": 5})

# Initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever_lancedb],
                                       weights=[0.4, 0.6])

# Example customer query
query = "which food needed for building strong bones and teeth ? which Vitamin & minerals importat for this?"


# Retrieve relevant documents/products
docs = ensemble_retriever.invoke(query)

llm = ChatOpenAI()

#if you want to use opensource models such as lama,mistral check this 
# https://github.com/lancedb/vectordb-recipes/blob/main/tutorials/chatbot_using_Llama2_&_lanceDB

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=ensemble_retriever)

query = "what nutrition needed for pregnant women  "
res =qa.invoke(query)
print(res)

# For an existing table, use get_table() instead of from_documents()
table = db.get_table("pandas_docs")

# Add new documents to the existing table
docsearch = LanceDB.add_documents(pages, embedding, table=table)