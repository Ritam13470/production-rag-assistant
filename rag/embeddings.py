from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class SafeGoogleEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=model_name
        )

    def embed_documents(self, texts):
        vectors = []

        for text in texts:
            vector = self.embedding_model.embed_query(text)
            vectors.append(vector)

        return vectors

    def embed_query(self, text):
        return self.embedding_model.embed_query(text)
