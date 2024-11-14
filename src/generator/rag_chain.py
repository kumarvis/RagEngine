from src.generator.create_prompt import get_prompt
from src.vector_db_builder.chroma import load_chroma_db
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama.llms import OllamaLLM

def get_rag_chain(config_dct):
    prompt = get_prompt()
    vectorstore = load_chroma_db(config_dct)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke("What is Appointee ?")
    print(len(docs))
    llm_model_name = config_dct["llm_model"]
    llm = OllamaLLM(model=llm_model_name)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain