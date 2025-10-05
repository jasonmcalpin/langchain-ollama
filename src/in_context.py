from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

"""
In-Context Learning Examples

In-context learning refers to the model's ability to adapt its behavior based on 
the context provided in the prompt, without requiring parameter updates or 
fine-tuning. The model learns from the context and examples within a single 
interaction.

Key characteristics:
- Context-dependent behavior adaptation
- Learning from provided examples and context
- No parameter updates required
- Immediate adaptation to new tasks or styles
- Context provides the "training data" within the prompt
"""

params = {
    "num_predict": 200,  # Sufficient for contextual responses
    "temperature": 0.7,  # Moderate creativity for adaptation
}

# Initialize Ollama with a specific model
llm = OllamaLLM(model="llama3.1:8b", **params)

# 1. Style Adaptation - Learning writing style from context
style_adaptation_prompt = """The following text demonstrates a specific writing style. Continue writing in the same style:

Context: {context}

Continue: {question}"""

# 2. Task Learning - Learning a new task from examples in context
task_learning_prompt = """Based on the following examples, perform the same task:

Context Examples:
{context}

Now perform the task: {question}"""

# 3. Domain Adaptation - Adapting to domain-specific language and concepts
domain_adaptation_prompt = """You are now working in a specific domain. Use the context to understand the domain language and concepts:

Domain Context: {context}

Question in this domain: {question}

Answer using appropriate domain language:"""

# 4. Format Learning - Learning output format from context
format_learning_prompt = """Follow the exact format shown in the context examples:

Format Examples:
{context}

Apply this format to: {question}"""

# Create prompt templates
style_template = PromptTemplate(template=style_adaptation_prompt, input_variables=["context", "question"])
task_template = PromptTemplate(template=task_learning_prompt, input_variables=["context", "question"])
domain_template = PromptTemplate(template=domain_adaptation_prompt, input_variables=["context", "question"])
format_template = PromptTemplate(template=format_learning_prompt, input_variables=["context", "question"])

# Create chains
style_chain = style_template | llm
task_chain = task_template | llm
domain_chain = domain_template | llm
format_chain = format_template | llm

# Run in-context learning examples
print("=== In-Context Learning Examples ===")

print("\n1. Style Adaptation - Learning Shakespearean Style:")
print("-" * 50)
style_result = style_chain.invoke({
    "context": "Hark! What light through yonder window breaks? 'Tis the east, and Juliet is the sun! Arise, fair sun, and kill the envious moon, who is already sick and pale with grief.",
    "question": "Describe a beautiful morning"
})
print(style_result)

print("\n2. Task Learning - Learning Translation Pattern:")
print("-" * 50)
task_result = task_chain.invoke({
    "context": "English: Hello -> Spanish: Hola\nEnglish: Thank you -> Spanish: Gracias\nEnglish: Goodbye -> Spanish: Adiós",
    "question": "English: Good morning"
})
print(task_result)

print("\n3. Domain Adaptation - Medical Domain:")
print("-" * 50)
domain_result = domain_chain.invoke({
    "context": "In cardiology, we often see patients with myocardial infarction (MI), which presents with chest pain, dyspnea, and elevated cardiac biomarkers. Treatment includes antiplatelet therapy, beta-blockers, and possible percutaneous coronary intervention (PCI).",
    "question": "What should be considered when a patient presents with acute chest pain?"
})
print(domain_result)

print("\n4. Format Learning - Structured Response Format:")
print("-" * 50)
format_result = format_chain.invoke({
    "context": "Question: What is Python?\nCategory: Programming\nDifficulty: Beginner\nAnswer: Python is a high-level programming language known for its simplicity and readability.\n\nQuestion: What is machine learning?\nCategory: AI/ML\nDifficulty: Intermediate\nAnswer: Machine learning is a subset of AI that enables computers to learn from data.",
    "question": "What is blockchain?"
})
print(format_result)

print("\n" + "=" * 55)