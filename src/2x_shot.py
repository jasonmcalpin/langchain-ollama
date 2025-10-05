from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

"""
Few-Shot Prompting Examples (2+ Examples)

Few-shot prompting involves providing multiple examples (typically 2 or more) 
before asking the model to perform the same task. This gives the model 
better context and understanding of the expected format and approach.

Key characteristics:
- Multiple clear examples provided (2 or more)
- Examples show consistent input-output format
- Demonstrates variety in the task
- Clear separation between examples and actual task
"""

params = {
    "num_predict": 150,  # Increased for better responses
    "temperature": 0.1,
}

# Initialize Ollama with a specific model (make sure it's pulled first)
llm = OllamaLLM(model="llama3.1:8b", **params)

# 1. Movie Review Classification (Few-shot)
movie_review_prompt = """Classify the following movie reviews as positive or negative.

Example 1:
Review: "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout."
Classification: Positive

Example 2:
Review: "I was extremely disappointed by this film. The plot was predictable and the acting felt wooden."
Classification: Negative

Example 3:
Review: "A masterpiece of cinema! Every scene was beautifully crafted and the story was deeply moving."
Classification: Positive

Now classify this review:
Review: {question}
Classification:"""

# 2. Emotion Classification (Few-shot)
emotion_classification_prompt = """Classify the emotion expressed in the following statements.

Example 1:
Statement: "I just won my first marathon!"
Emotion: Joy

Example 2:
Statement: "I can't believe I lost my keys again."
Emotion: Frustration

Example 3:
Statement: "My best friend is moving to another country."
Emotion: Sadness

Now classify the emotion:
Statement: {question}
Emotion:"""

# 3. English to Spanish Translation (Few-shot)
translation_prompt = """Translate the following English text to Spanish.

Example 1:
English: "Good morning, how are you today?"
Spanish: "Buenos días, ¿cómo estás hoy?"

Example 2:
English: "Thank you very much for your help."
Spanish: "Muchas gracias por tu ayuda."

Example 3:
English: "Where is the nearest restaurant?"
Spanish: "¿Dónde está el restaurante más cercano?"

Now translate:
English: {question}
Spanish:"""

# 4. Text Summarization (Few-shot)
summarization_prompt = """Summarize the following paragraphs in one sentence.

Example 1:
Paragraph: "Renewable energy sources like solar and wind power are becoming increasingly important as alternatives to fossil fuels. They produce clean energy without harmful emissions and are becoming more cost-effective each year."
Summary: Renewable energy sources like solar and wind are growing as clean, cost-effective alternatives to fossil fuels.

Example 2:
Paragraph: "Artificial intelligence is transforming various industries by automating complex tasks, improving decision-making processes, and enabling new innovations. Companies are investing heavily in AI technologies to stay competitive."
Summary: AI is revolutionizing industries through automation and innovation, driving significant corporate investment.

Now summarize this paragraph:
Paragraph: {question}
Summary:"""

# Create prompt templates
movie_review = PromptTemplate(template=movie_review_prompt, input_variables=["question"])
emotion_classification = PromptTemplate(template=emotion_classification_prompt, input_variables=["question"])
translation = PromptTemplate(template=translation_prompt, input_variables=["question"])
summarization = PromptTemplate(template=summarization_prompt, input_variables=["question"])

# Create chains
movie_chain = movie_review | llm
emotion_chain = emotion_classification | llm
translation_chain = translation | llm
summarization_chain = summarization | llm

# Run the chains with few-shot examples
print("=== Few-Shot Prompting Examples (2+ Examples) ===")

print("\n1. Movie Review Classification:")
print("-" * 40)
movie_result = movie_chain.invoke({"question": "The cinematography was breathtaking and the soundtrack perfectly complemented every emotional moment. A truly remarkable film!"})
print(movie_result)

print("\n2. Emotion Classification:")
print("-" * 40)
emotion_result = emotion_chain.invoke({"question": "That movie was so scary I had to cover my eyes."})
print(emotion_result)

print("\n3. English to Spanish Translation:")
print("-" * 40)
translation_result = translation_chain.invoke({"question": "I would like to order a coffee with milk and two sugars, please."})
print(translation_result)

print("\n4. Text Summarization:")
print("-" * 40)
summarization_result = summarization_chain.invoke({"question": "Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil and gas, which produces heat-trapping gases. The consequences of climate change include more frequent and severe droughts, storms, and heat waves, rising sea levels, melting glaciers, and warming oceans which can directly impact biodiversity, agriculture, and human health."})
print(summarization_result)

print("\n" + "=" * 55)