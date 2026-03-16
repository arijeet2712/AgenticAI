
import os
import mlflow
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ.get("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

internal_docs = [
    Document(page_content="""
        Azure Policy: Cloud Spend Governance v2.3
        All Azure resources must be tagged: CostCenter, Environment, Owner.
        Monthly cloud spend over $50,000 requires VP Finance approval.
        Orphaned resources (no activity >30 days) are auto-flagged for deletion.
    """, metadata={"source": "cloud_policy_v2.3"}),
    Document(page_content="""
        Azure Architecture Standard: Networking
        All production workloads must use Private Endpoints.
        Hub-and-spoke topology is required for enterprise subscriptions.
    """, metadata={"source": "arch_network_2024"}),
    Document(page_content="""
        Data Classification Policy
        Level 4 (Restricted): Encryption at rest + in transit + Customer Managed Keys.
        PII and financial data are always classified as Level 4.
    """, metadata={"source": "data_classification_2024"}),
]

chunks    = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(internal_docs)
vs        = FAISS.from_documents(chunks, embeddings)
retriever = vs.as_retriever(search_kwargs={"k": 3})

llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    temperature=0.0,
    max_tokens=1024,
)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are an enterprise Azure compliance advisor.
Answer ONLY from the provided internal policy documents.
If the answer is not in the context, say so - do not invent.
Always cite the source document name.

Context:
{context}"""),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content.strip()}"
        for d in docs
    )

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

mlflow.models.set_model(chain)
