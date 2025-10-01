from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

params = {
    "num_predict": 512,
    "temperature": 0.5,
}


# Initialize Ollama with a specific model (make sure it's pulled first)
llm = OllamaLLM(model="llama3.1:8b", **params)

prompt = """When I was 6, my sister was half of my age. Now I am 70, what age is my sister?

            Provide three independent calculations and explanations, then determine the most consistent result.

"""

# Run the chain
print("\n-------------------\n")

result = llm.invoke(prompt)
print(result)
print("\n-------------------\n")

GPS:Respawn Pod - 72805422979530382:21129.7609862895:-53511.4648438818:22064.7089844985:#FF75C9F1:

GPS:base:29323.4683178102:-6530.57889182333:52761.0405101092:#FF75C9F1: