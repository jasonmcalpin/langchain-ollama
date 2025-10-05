from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize Ollama with a specific model (make sure it's pulled first)
llm = OllamaLLM(model="llama3.1:8b")

# Simple query
# response = llm.invoke("The Eiffel Tower is located in Berlin?")
# print(response)

# Using a prompt template
template = """Classify the following statement as true or false:  {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Create a chain
chain = prompt | llm

# Run the chain
result = chain.invoke({"question": "is photosynthesis a mechanical process?"})
print(result)