from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

"""
1-Shot Prompting Examples

1-shot prompting involves providing exactly one example or demonstration 
before asking the model to perform the same task. This helps the model 
understand the expected format and approach.

Key characteristics:
- One clear example provided
- Example shows input-output format
- Task instruction followed by the example
- Clear separation between example and actual task
"""

params = {
    "num_predict": 100,  # Increased for better responses
    "temperature": 0.1,
}
# Initialize Ollama with a specific model (make sure it's pulled first)
llm = OllamaLLM(model="llama3.1:8b", **params)

# 1. Movie Review Classification (1-shot)
movie_review_prompt = """Classify the following movie review as positive or negative.

Example:
Review: "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout."
Classification: Positive

Now classify this review:
Review: {question}
Classification:"""

# 2. Climate Change Paragraph Summarization (1-shot)
climate_change_prompt = """Summarize the following paragraph in one sentence.

Example:
Paragraph: "Renewable energy sources like solar and wind power are becoming increasingly important as alternatives to fossil fuels. They produce clean energy without harmful emissions and are becoming more cost-effective each year."
Summary: Renewable energy sources like solar and wind are growing as clean, cost-effective alternatives to fossil fuels.

Now summarize this paragraph:
Paragraph: {question}
Summary:"""

# 3. English to Spanish Translation (1-shot)
translation_prompt = """Translate the following English text to Spanish.

Example:
English: "Good morning, how are you today?"
Spanish: "Buenos días, ¿cómo estás hoy?"

Now translate:
English: {question}
Spanish:"""

# 4. True/False Classification (1-shot)
true_false_prompt = """Classify the following statement as true or false.

Example:
Statement: "Water boils at 100 degrees Celsius at sea level."
Answer: True

Now classify:
Statement: {question}
Answer:"""


one_shot_translation_prompt = """Here is an example of translating a sentence from English to French:

            English: “How is the weather today?”
            French: “Comment est le temps aujourd'hui?”
            
            Now, translate the following sentence from English to French:

            English: {question}
"""

# Create prompt templates
movie_review = PromptTemplate(template=movie_review_prompt, input_variables=["question"])
climate_change = PromptTemplate(template=climate_change_prompt, input_variables=["question"])
translation = PromptTemplate(template=translation_prompt, input_variables=["question"])
true_false = PromptTemplate(template=true_false_prompt, input_variables=["question"])

# Create chains
movie_chain = movie_review | llm
climate_chain = climate_change | llm
translation_chain = translation | llm
true_false_chain = true_false | llm

# Run the chains with 1-shot examples
print("=== 1-Shot Prompting Examples ===")

print("\n1. Movie Review Classification:")
print("-" * 40)
movie_result = movie_chain.invoke({"question": "I was extremely disappointed by this film. The plot was predictable, the acting was wooden, and the special effects looked cheap. I can't recommend this to anyone."})
print(movie_result)

print("\n2. Climate Change Summarization:")
print("-" * 40)
climate_result = climate_chain.invoke({"question": "Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil and gas, which produces heat-trapping gases. The consequences of climate change include more frequent and severe droughts, storms, and heat waves, rising sea levels, melting glaciers, and warming oceans which can directly impact biodiversity, agriculture, and human health."})
print(climate_result)

print("\n3. English to Spanish Translation:")
print("-" * 40)
translation_result = translation_chain.invoke({"question": "I would like to order a coffee with milk and two sugars, please."})
print(translation_result)

print("\n4. True/False Classification:")
print("-" * 40)
true_false_result = true_false_chain.invoke({"question": "Photosynthesis is a mechanical process."})
print(true_false_result)

print("\n" + "=" * 50)
