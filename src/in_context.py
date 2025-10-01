def warn (*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import LLMChain

def llm_model(prompt_txt, params=None):
    
    model = "deepseek-r1:7b"

    default_params = {
        "temperature": 0.5,
        "top_p": 0.2,
        "top_k": 1,
        "num_predict": 256  # Ollama equivalent of max_new_tokens
    }

    if params:
        default_params.update(params)

    # Create Ollama LLM - assumes Ollama is running locally on default port 11434
    ollama_llm = OllamaLLM(
        model=model,
        base_url="http://localhost:11434",  # Default Ollama URL
        temperature=default_params.get("temperature", 0.5),
        top_p=default_params.get("top_p", 0.2),
        top_k=default_params.get("top_k", 1),
        num_predict=default_params.get("num_predict", 256)
    )
    
    response = ollama_llm.invoke(prompt_txt)
    return response


params = {
    "num_predict": 128,  # Ollama equivalent of max_new_tokens
    "temperature": 0.5,
    "top_p": 0.2,
    "top_k": 1
}

prompt = "The wind is "

# Getting a reponse from the model with the provided prompt and new parameters
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")