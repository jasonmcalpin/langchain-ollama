from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

params = {
    "num_predict": 20,
    "temperature": 0.1,
}
# Initialize Ollama with a specific model (make sure it's pulled first)
llm = OllamaLLM(model="llama3.1:8b", **params)

# 1. Prompt for Movie Review Classification
movie_review_prompt = """Classify the following movie review as positive or negative: {question}"""

# 2. Prompt for Climate Change Paragraph Summarization
climate_change_prompt = """Summarize the following paragraph on climate change: {question}"""

# 3. Prompt for English to Spanish Translation
translation_prompt = """Translate the following English text to Spanish without showing any examples: {question}"""

# Using a prompt template
template = """Classify the following statement as true or false:  {question}

Answer: Let's think step by step."""


one_shot_translation_prompt = """Here is an example of translating a sentence from English to French:

            English: “How is the weather today?”
            French: “Comment est le temps aujourd'hui?”
            
            Now, translate the following sentence from English to French:

            English: {question}
"""

prompt = PromptTemplate(template=template, input_variables=["question"])

movie_review = PromptTemplate(template=movie_review_prompt, input_variables=["question"])
climate_change = PromptTemplate(template=climate_change_prompt, input_variables=["question"])
translation = PromptTemplate(template=translation_prompt, input_variables=["question"])
one_shot_translation = PromptTemplate(template=one_shot_translation_prompt, input_variables=["question"])

# Create a chain
chain = prompt | llm
movie_chain = movie_review | llm
climate_chain = climate_change | llm
translation_chain = translation | llm
one_shot_translation_chain = one_shot_translation | llm

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
print("\n-------------------\n")

one_shot_translation_result = one_shot_translation_chain.invoke({"question": "Where is the nearest supermarket?"})
print(one_shot_translation_result)
print("\n-------------------\n")
