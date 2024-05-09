# Parts of this code taken from reference: https://github.com/pixegami/rag-tutorial-v2

from langchain_community.embeddings.ollama import OllamaEmbeddings

# from langchain_community.embeddings.bedrock import BedrockEmbeddings


def get_embedding_function(myModel):
    # embeddings = BedrockEmbeddings(
    # credentials_profile_name="default", region_name="us-east-1"
    # ) # use Amazon Web Server
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    embeddings = OllamaEmbeddings(model=myModel)  # use local server with Ollama
    return embeddings
