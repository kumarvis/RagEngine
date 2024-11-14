from src.generator.rag_chain import get_rag_chain
from config.load_config import load_yaml_config

def answer_query(config_path):
    config_dct = load_yaml_config(config_path)
    # Initialize the RAG (Retrieval-Augmented Generation) chain
    rag_chain = get_rag_chain(config_dct)
    
    # Start an interactive loop for queries
    while True:
        # Get the query from the user
        query = "What is Appointee ?" #input("Enter your query (or type 'bye' to exit): ")

        if query.lower() == "bye":
            print("Goodbye!")
            break
        
        # Invoke the RAG chain with the user's query and print the response
        response = rag_chain.invoke(query)
        print("Response:", response)
        print("Next")
    


if __name__ == "__main__":
    config_path = "config/config.yaml"
    answer_query(config_path)
