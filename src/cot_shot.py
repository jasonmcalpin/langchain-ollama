from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

"""
Chain-of-Thought (CoT) Prompting Examples

Chain-of-Thought prompting encourages the model to show its reasoning process 
step-by-step before arriving at a final answer. This approach helps with 
complex reasoning tasks and makes the model's thinking process transparent.

Key characteristics:
- Explicit step-by-step reasoning
- "Let's think step by step" or similar phrases
- Breaking down complex problems into smaller parts
- Showing intermediate steps and calculations
- Clear logical progression to the final answer
"""

params = {
    "num_predict": 300,  # Increased for detailed reasoning
    "temperature": 0.3,  # Lower temperature for more focused reasoning
}

# Initialize Ollama with a specific model (make sure it's pulled first)
llm = OllamaLLM(model="llama3.1:8b", **params)

# 1. Math Word Problem (CoT)
math_problem_prompt = """Solve this step by step:

When I was 6, my sister was half of my age. Now I am 70, what age is my sister?

Let's think step by step:

Step 1: Determine the age difference
Step 2: Apply the age difference to the current situation
Step 3: Calculate the final answer

Solution:"""

# 2. Logic Reasoning (CoT)
logic_reasoning_prompt = """Answer this logic question by thinking through it step by step:

{question}

Let's think step by step:

Step 1: Identify what we know
Step 2: Determine what we need to find
Step 3: Apply logical reasoning
Step 4: Reach a conclusion

Answer:"""

# 3. Reading Comprehension with Reasoning (CoT)
reading_comprehension_prompt = """Read the following passage and answer the question by explaining your reasoning:

{question}

Let's think step by step:

Step 1: Identify key information from the passage
Step 2: Understand what the question is asking
Step 3: Connect the relevant information
Step 4: Formulate the answer

Answer:"""

# 4. Problem Solving (CoT)
problem_solving_prompt = """Solve this problem by breaking it down into steps:

{question}

Let's think step by step:

Step 1: Understand the problem
Step 2: Identify the approach
Step 3: Work through the solution
Step 4: Verify the answer

Solution:"""

# Create prompt templates
math_template = PromptTemplate(template=math_problem_prompt, input_variables=[])
logic_template = PromptTemplate(template=logic_reasoning_prompt, input_variables=["question"])
reading_template = PromptTemplate(template=reading_comprehension_prompt, input_variables=["question"])
problem_template = PromptTemplate(template=problem_solving_prompt, input_variables=["question"])

# Create chains
math_chain = math_template | llm
logic_chain = logic_template | llm
reading_chain = reading_template | llm
problem_chain = problem_template | llm

# Run the chains with Chain-of-Thought prompting
print("=== Chain-of-Thought Prompting Examples ===")

print("\n1. Math Word Problem:")
print("-" * 40)
math_result = math_chain.invoke({})
print(math_result)

print("\n2. Logic Reasoning:")
print("-" * 40)
logic_result = logic_chain.invoke({"question": "All cats are mammals. Fluffy is a cat. Is Fluffy a mammal?"})
print(logic_result)

print("\n3. Reading Comprehension:")
print("-" * 40)
reading_result = reading_chain.invoke({"question": "The Amazon rainforest is often called the 'lungs of the Earth' because it produces about 20% of the world's oxygen. However, recent studies show that the Amazon actually consumes almost as much oxygen as it produces through respiration. The real value of the Amazon lies in its role as a carbon sink, absorbing billions of tons of carbon dioxide from the atmosphere. Question: What is the primary environmental benefit of the Amazon rainforest according to recent studies?"})
print(reading_result)

print("\n4. Problem Solving:")
print("-" * 40)
problem_result = problem_chain.invoke({"question": "A farmer has 100 meters of fencing and wants to create a rectangular pen with the maximum possible area. What should be the dimensions of the pen?"})
print(problem_result)

print("\n" + "=" * 50)