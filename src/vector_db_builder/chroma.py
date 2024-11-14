
from langchain.vectorstores import Chroma
from src.vector_db_builder.document_splitter import split_documents
from src.loaders.main_load import Loader
import fs_utils.file_system_utility as fsutils
from src.vector_db_builder.embeddings import get_embedding_model

def create_chroma_db(doc_path, config_dct):
    doc_name, ext = fsutils.get_file_name_and_extension(doc_path)
    obj_loader = Loader()
    documents = obj_loader.load(doc_path)
    splits = split_documents(documents)
    embedding_model = get_embedding_model(config_dct)

    db_path = config_dct["vector_db_path"]
    vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding_model,
    persist_directory=db_path)  # specify directory to save vector data
    vectorstore.persist()

def load_chroma_db(config_dct):
    embedding_model = get_embedding_model(config_dct)
    db_path = config_dct["vector_db_path"]
    vectorstore = Chroma(
    persist_directory=db_path,
    embedding_function=embedding_model)
    return vectorstore




