from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pinecone.grpc import PineconeGRPC
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain_anthropic import ChatAnthropic
import os
import yaml
import time


def get_config(config: str):
    # Reads a YAML configuration file.
    with open(f"{config}.yaml", "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


config = get_config("config")
os.environ["PINECONE_API_KEY"] = config['api-key']
os.environ["ANTHROPIC_API_KEY"] = config['sonnet-key']

model_name = 'multilingual-e5-large'
embeddings = PineconeEmbeddings(
    model=model_name,
    pinecone_api_key=os.environ.get('PINECONE_API_KEY')
)

loader = PyPDFLoader("data/Caide.pdf",)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
chunks = text_splitter.split_documents(documents)

pc = PineconeGRPC(api_key=config['api-key'])
cloud = 'aws'
region = 'us-east-1'
# spec = ServerlessSpec(cloud=cloud, region=region)
index_name = "rag-test"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embeddings.dimension,
        metric="cosine",
        spec={'serverless': {'cloud': 'aws', 'region': 'us-east-1'}}
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)


namespace = "internal_guides"
# namespace = "wondervector5000"

docsearch = PineconeVectorStore.from_documents(
    documents=chunks,
    index_name=index_name,
    embedding=embeddings,
    namespace=namespace
)

index = pc.Index(index_name)
for ids in index.list(namespace=namespace):
    query = index.query(
        id=ids[0],
        namespace=namespace,
        top_k=1,
        include_values=True,
        include_metadata=True
    )
    print(query)
    print("\n")

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
retriever = docsearch.as_retriever()

openai_llm = ChatOpenAI(
    openai_api_key=config['open-ai-key'],
    model_name='gpt-4o-mini',
    temperature=0.0
)

claude_llm = ChatAnthropic(
    temperature=0.0,
    api_key=os.environ.get('ANTHROPIC_API_KEY'),
    model_name="claude-3-opus-20240229"
)

combine_docs_chain = create_stuff_documents_chain(
    openai_llm, retrieval_qa_chat_prompt
)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

query1 = "How can I update the tone of voice?"
answer1_without_knowledge = openai_llm.invoke(query1)
print("Query 1:", query1)
print("\nAnswer without knowledge:\n\n", answer1_without_knowledge.content)
print("\n")

# openai_llm.invoke("start the LLM")
query2 = "Where I can see all my email thread?"

start_time = time.time()
answer2_with_knowledge = retrieval_chain.invoke({"input": query2})
duration = time.time() - start_time

print("\nAnswer with knowledge:\n\n", answer2_with_knowledge['answer'])
print("\nContext Used:\n\n", answer2_with_knowledge['context'])
print("\n")
print(f'time : {duration}')

pc.delete_index(index_name)