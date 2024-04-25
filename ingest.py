from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    # print("text--------------",texts)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # print("embeddings-------------",dir(embeddings))
    db = FAISS.from_documents(texts, embeddings)
    # print("Helpful statements------------=",dir(db.index))
    # print("Another one--------------------",dir(db))
    # vector_Emb = db.index.reconstruct_n()
    # print(len(vector_Emb),vector_Emb)
    # print("db------ SEE HERE --------", db)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()

