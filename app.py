import os
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
)
from langchain_pinecone import PineconeVectorStore
from src.prompt import system_prompt

app = Flask(__name__)

load_dotenv()

# Environment Variables
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not PINECONE_API_KEY or not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError(
        "Missing PINECONE_API_KEY or HUGGINGFACEHUB_API_TOKEN in .env file."
    )

index_name = "medicalbot"

# 1. Load Embeddings & Vector Store Retriever
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name, embedding=embeddings
)
retriever = docsearch.as_retriever(
    search_type="similarity", search_kwargs={"k": 3}
)

# 2. LLM Setup
llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.5,
    task="text-generation",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)
llm = ChatHuggingFace(llm=llm_endpoint)

# 3. Prompt Setup
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 4. Construct LCEL RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# 5. Flask Routes
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"User Question: {msg}")

    # LCEL returns a string directly (StrOutputParser)
    response = rag_chain.invoke(msg)
    print(f"Bot Response: {response}")

    return str(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)