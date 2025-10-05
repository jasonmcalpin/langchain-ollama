from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

"""
0-Shot Prompting Examples

0-shot prompting involves asking the model to perform a task without providing 
any examples or demonstrations. The model relies solely on its pre-trained 
knowledge and the task description to generate responses.

Key characteristics:
- No examples provided in the prompt
- Clear task instructions
- Direct question/instruction format
"""

llm = OllamaLLM(model="llama3.1:8b")

# 1. Prompt for Movie Review Classification (0-shot)
movie_review_prompt = """Classify the following movie review as positive or negative: {question}"""

# 2. Prompt for Climate Change Paragraph Summarization (0-shot)
climate_change_prompt = """Summarize the following paragraph on climate change: {question}"""

# 3. Prompt for English to Spanish Translation (0-shot)
translation_prompt = """Translate the following English text to Spanish: {question}"""

# 4. Prompt for True/False Classification (0-shot)
true_false_prompt = """Classify the following statement as true or false: {question}

Answer:"""

prompt = PromptTemplate(template=true_false_prompt, input_variables=["question"])

movie_review = PromptTemplate(template=movie_review_prompt, input_variables=["question"])
climate_change = PromptTemplate(template=climate_change_prompt, input_variables=["question"])
translation = PromptTemplate(template=translation_prompt, input_variables=["question"])

# Create a chain
chain = prompt | llm
movie_chain = movie_review | llm
climate_chain = climate_change | llm
translation_chain = translation | llm

# Run the chain
print("\n-------------------\n")

result = chain.invoke({"question": "is photosynthesis a mechanical process?"})
print(result)
print("\n-------------------\n")

movie_result = movie_chain.invoke({"question": "I was extremely disappointed by this film. The plot was predictable, the acting was wooden, and the special effects looked cheap. I can't recommend this to anyone."})
print(movie_result)
print("\n-------------------\n")

climate_result = climate_chain.invoke({"question": "Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil and gas, which produces heat-trapping gases. The consequences of climate change include more frequent and severe droughts, storms, and heat waves, rising sea levels, melting glaciers, and warming oceans which can directly impact biodiversity, agriculture, and human health."})
print(climate_result)
print("\n-------------------\n")

translation_result = translation_chain.invoke({"question": "I would like to order a coffee with milk and two sugars, please."})
print(translation_result)

