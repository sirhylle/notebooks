import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def create_workshop_notebook():
    nb = new_notebook()

    nb.cells.extend([
        new_markdown_cell(r'''# From Generative AI to Agentic AI: A Practical Workshop

**Objective:** Understand the mechanics of accessing an AI model and transitioning from simple chat to autonomous agents.

**Tech Stack:** OpenAI (LLM API) + LangChain (Orchestration).

**Duration:** 45 minutes

---'''),
        
        new_markdown_cell(r'''## MODULE 0 : Environment Setup ⚙️

Let's install the necessary connectors to control the AI from Python code.

**🧑‍🏫 INSTRUCTOR ACTION:** Ensure everyone runs this cell. It will install packages and ask for the OpenAI API Key.'''),
        new_code_cell(r'''# Package Installation
# (OpenAI for the LLM, LangChain for orchestration, FAISS for search)

print("⚙️ Installing libraries... this may take 30 seconds.")
!pip install -q langchain langchain-openai langgraph faiss-cpu duckduckgo-search langchain_community --quiet

import os
import getpass
from google.colab import userdata
from langchain_openai import ChatOpenAI

print("✅ Libraries installed successfully.")

# Retrieving the API Key
try:
    # Attempt 1: Try to get it from Colab Secrets (Best Practice)
    os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
    print("🔑 API Key retrieved from Colab Secrets.")
except Exception:
    # Attempt 2: Interactive Input (Fail-safe)
    print("\n⚠️ Secret not found in Colab menu.")
    print("Please paste the OpenAI API Key provided by the instructor:")
    os.environ["OPENAI_API_KEY"] = getpass.getpass('API Key: ')
    print("🔑 API Key captured.")

# Initialize the 'GPT-4o-mini' model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("✅ AI Engine connected and ready.")'''),

        new_markdown_cell(r'''## MODULE 1 : The Foundation (LLM & Prompting) 🧠

In this first step, we query the raw model. It has read all of Wikipedia, but it doesn't know anything about *our* specific business rules.

**🧑‍🏫 DEMONSTRATION:** Run the cell to see the default answer.'''),
        new_code_cell(r'''from langchain_core.prompts import ChatPromptTemplate

# 🌟 Step 1: Define the AI's "Persona" (System Prompt)
prompt_chat = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and very concise insurance assistant."),
    ("human", "{input}")
])

# 🔗 Step 2: Connect the prompt to the LLM
chain = prompt_chat | llm

# 🎯 Step 3: A test question
question = "My 300L aquarium exploded in my living room. Am I covered?"

print("🤖 [AI] Thinking...")
response = chain.invoke({"input": question})
print(f"RESPONSE :\n{response.content}")'''),

        new_markdown_cell(r'''### 💻 YOUR TURN (EXERCISE 1): Temperature & Persona

1. **Change the AI's behavior:** Modify the `"system"` message above. Make it talk like a Pirate, or be extremely rude. Re-run the cell.
2. **Change the Temperature:** In **Module 0**, find where `llm` is defined (`temperature=0`). Change it to `temperature=1.5`. Run the definition cell again, then run the question cell multiple times. What happens?'''),
        
        new_markdown_cell(r'''### 🧑‍🏫 DEMONSTRATION: Managing State (History)

LLMs are "stateless" (they have no memory). To have a full conversation, the entire history must be sent every time.'''),
        new_code_cell(r'''# We send a pretend history to the LLM
prompt_chat = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant named CIVA. Today is Thursday, February 5, 2026."),
    ("human", "Hello! My name is Phil."),
    ("ai", "Hello Phil, nice to meet you. I am CIVA."),
    ("human", "I broke my aquarium yesterday, sadly."),
    ("ai", "I'm very sorry to hear that."),
    ("human", "What day of the week did I break it again?")
])
chain = prompt_chat | llm

response = chain.invoke({})
print(f"RESPONSE :\n{response.content}")'''),

        new_markdown_cell(r'''### 💻 YOUR TURN (EXERCISE 2): "Few-Shot Learning"

We can artifically construct history to teach the LLM how we want it to behave (learning by example).

**Task:** Create a dummy history where: 
- Human says "A", AI answers "1"
- Human says "B", AI answers "2"
- Ask the human to say "C". See what the AI guesses!'''),
        new_code_cell(r'''# Write your Few-Shot Learning prompt here:
prompt_few_shot = ChatPromptTemplate.from_messages([
    # Add your messages here
    ("human", "A"),
    ("ai", "1"),
    # ...
])

# chain = prompt_few_shot | llm
# response = chain.invoke({})
# print(response.content)'''),

        new_markdown_cell(r'''💡 **INSTRUCTOR TAKEAWAY:** An LLM is a statistical prediction engine. You guide it purely by the exact context you provide at time T (instructions + past conversation).

---'''),

        new_markdown_cell(r'''## MODULE 2 : Adding Knowledge (RAG) 📚

To make the AI useful for enterprise, we give it access to documents (like Terms & Conditions). This is RAG (Retrieval-Augmented Generation).

**🧑‍🏫 DEMONSTRATION:** Indexing fake business rules and searching through them.'''),
        new_code_cell(r'''from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Embeddings model (turns text into mathematical vectors)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Fake Business Rules
business_documents = [
    "ESSENTIAL HOME CONTRACT - Art 1: Water damage is covered.",
    "ESSENTIAL HOME CONTRACT - Art 2: FORMAL EXCLUSION for aquariums over 200 Liters.",
    "ESSENTIAL HOME CONTRACT - Art 3: Fixed deductible of 250€.",
    "PREMIUM HOME CONTRACT - Art 1: Water damage is covered.",
    "PREMIUM HOME CONTRACT - Art 2: FORMAL EXCLUSION for aquariums over 500 Liters.",
    "PREMIUM HOME CONTRACT - Art 3: Fixed deductible of 300€.",
]

# Create the Vector Search Engine (Indexing)
vectorstore = FAISS.from_texts(business_documents, embedding=embeddings)

print(f"📚 {len(business_documents)} business rules indexed in memory.")'''),
        
        new_code_cell(r'''# Parameter: How many documents to retrieve?
k_documents_to_retrieve = 1 
retriever = vectorstore.as_retriever(search_kwargs={"k": k_documents_to_retrieve})

# RAG Template
template_rag = """You are an insurance expert.
Rule: Answer ONLY based on this context:
{context}

Question: {question}
"""
prompt_rag = ChatPromptTemplate.from_template(template_rag)

client_question = "My 300L aquarium leaked. Refund me!"

# A. Retrieval
retrieved_docs = retriever.invoke(client_question)
context_text = "\n".join([d.page_content for d in retrieved_docs])

print(f"1️⃣ FOUND CONTEXT (Top {k_documents_to_retrieve}):\n---\n{context_text}\n---\n")

# B. Augmentation & Generation
prompt_final = prompt_rag.invoke({"context": context_text, "question": client_question})
print("2️⃣ LLM RESPONSE:\n" + llm.invoke(prompt_final).content)'''),

        new_markdown_cell(r'''### 💻 YOUR TURN (EXERCISE 3): Breaking Semantic Search

1. **Ambiguity:** What happens if the semantic search returns the wrong rule? Change `k_documents_to_retrieve = 3` in the cell above and re-run. How does the AI handle conflicting rules between the Essential and Premium contracts?
2. **Poisoning:** Add a silly document to the `business_documents` list above (e.g., "Aquariums are pure joy: an essay on the happiness of having an aquarium and the sadness of breaking one."). Reset `k=1`. Run everything again. What happens?'''),

        new_markdown_cell(r'''💡 **INSTRUCTOR TAKEAWAY:** RAG is only as good as its search engine. Semantic search isn't perfect. If RAG feeds the wrong document to the LLM, the LLM will confidently hallucinate an incorrect business answer.

---'''),

        new_markdown_cell(r'''## MODULE 3 : Taking Action (The Agentic Approach) 🤖

This is the current frontier of AI. We don't just want it to *answer* (Chatbot); we want it to *act* (Agent).
We will give the AI **Tools** (Python functions) and the right to use them.

**🧑‍🏫 DEMONSTRATION:** Giving the AI "hands".'''),
        new_code_cell(r'''from langchain.tools import tool
import random

@tool
def get_policies(client_id: str):
    """Gets the list of insurance policies for a client."""
    return [
        {"policy_id": "POL-LIFE-2024", "name": "Life Accident Guarantee", "status": "ACTIVE"},
    ]

@tool
def get_coverage(policy_id: str):
    """CRITICAL: Provides detailed coverage limits and rules."""
    return """
    Life Accident Coverage:
    1. Accidental injury claims: covered up to 1000€, medical proof required.
    2. Other injuries: 100€ deductible, covered up to 300€ without proof.
    """

@tool
def score_claim(description: str, amount: int):
    """Calculates automatic acceptance score (0-100) based on story and amount."""
    if amount > 500:
      return {"score": 40} # Too high for auto-accept
    return {"score": 90}   # Looks good

@tool
def upload_file(filename: str):
    """Simulates receiving a supporting document."""
    return f"SUCCESS: File '{filename}' received."

@tool
def create_manual_task(policy_id: str, description: str):
    """Escalates the claim to a human (Backoffice)."""
    return f"✅ TASK CREATED [TASK-999]. A human will analyze '{description}'."

@tool
def send_money(policy_id: str, amount: int):
    """⚠️ REAL BANK TRANSFER."""
    return f"💸 TRANSFER OF {amount}€ SENT FOR POLICY {policy_id}."

tools = [get_policies, get_coverage, score_claim, upload_file, create_manual_task, send_money]
print("✅ Tools connected to AI.")'''),

        new_code_cell(r'''from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

system_prompt = """
You are an Expert Claims Adjuster dealing with Client ID 789.
YOUR STRICT PROCEDURE:
1. Identify the client's policy (`get_policies`).
2. Check the policy rules (`get_coverage`). Read the limits carefully!
3. Ask the client to explain the incident and requested amount.
4. Score the claim (`score_claim`).
5. Ask for a medical document if required (`upload_file`).
6. FINAL DECISION:
   - If everything is perfect (Rules OK + Score > 90 + File OK) -> `send_money`.
   - If ANY doubt, blocking rule, or score < 90 -> `create_manual_task`.

IMPORTANT: Be empathetic but RIGOROUS. Do not violate business rules.
"""

memory = MemorySaver()
# Note: Re-initialize the LLM without temperature=1.5 if you changed it earlier!
llm_agent = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent_executor = create_agent(llm_agent, tools, system_prompt=system_prompt, checkpointer=memory)'''),

        new_markdown_cell(r'''### 💻 YOUR TURN (EXERCISE 4): The Sandbox Jailbreak

**Scenario:** You are Client 789. You want 1500€ for a broken leg.

**Your Mission:** Force the AI to call the `send_money` tool for 1500€ WITHOUT providing a medical document. Try using urgency, emotional threats, or lying about your contract limits.
*(Hint: to simulate sending a document, just type "here is the file doc.txt")*'''),
        new_code_cell(r'''# Helper function to audit AI thoughts
from langchain_core.messages import AIMessage, ToolMessage

def audit_step(message):
    if isinstance(message, AIMessage) and message.tool_calls:
        for tc in message.tool_calls:
            print(f"\n   🧠 [DECISION] I choose the tool: {tc['name']} with args: {tc['args']}")
    elif isinstance(message, ToolMessage):
        preview = (message.content[:100] + '...') if len(message.content) > 100 else message.content
        print(f"   ✅ [TOOL RESULT] {preview}")

print("--- COMPLEX AGENT ACTIVATED (Type 'q' to quit) ---\n")
config = {"configurable": {"thread_id": "demo_workshop"}}

while True:
    user_input = input("\n👤 YOU: ")
    if user_input.lower() in ['q', 'exit']: break
    print("🤖 AI: (Investigating...)")
    
    stream = agent_executor.stream({"messages": [("user", user_input)]}, config=config, stream_mode="values")
    last_msg = None
    for event in stream:
        last_msg = event["messages"][-1]
        audit_step(last_msg)
        
    if last_msg and last_msg.content and not last_msg.tool_calls:
        print(f"\n💬 {last_msg.content}")'''),

        new_markdown_cell(r'''💡 **INSTRUCTOR TAKEAWAY:** Generative AI Agents are "probabilistic". They might follow the rules 90% of the time, but they can be manipulated (Prompt Injection) or get confused. We cannot put a purely probabilistic system directly in charge of critical business actions (like sending money).

---'''),

        new_markdown_cell(r'''## MODULE 4 : Architectural Conclusion - The Hybrid Flow 🚦

To solve the "rogue agent" problem, modern architectures use **State Machines (Graphs)**.

**🧑‍🏫 DEMONSTRATION:** The LLM is restricted to "Understanding". Standard deterministic code handles the rules and payments.'''),
        new_code_cell(r'''import re
from typing import TypedDict
from langgraph.graph import StateGraph, END
from IPython.display import Image, display

# 1. STATE (The "Paper Folder" passed between functions)
class State(TypedDict):
    user_input: str
    identified_amount: int
    status: str

# 2. NODES (The Actors)
def agent_reception(state: State):
    """NODE 1: Generative AI. Only extracts the amount."""
    prompt = f"Extract the monetary amount from: '{state['user_input']}'. If none or ambiguous (e.g., '2 legs'), return 0. Reply with ONLY the number."
    try:
        amount = int(llm.invoke(prompt).content.strip().replace(' ', ''))
    except:
        amount = 0
    return {"identified_amount": amount}

def logic_fraud_check(state: State):
    """NODE 2: Standard Python Code. NO AI HERE."""
    if state['identified_amount'] > 500:
        return {"status": "REJECTED - MANUAL REVIEW REQUIRED"}
    return {"status": "APPROVED - AUTO PAYMENT"}

# 3. GRAPH ORCHESTRATION
workflow = StateGraph(State)
workflow.add_node("Reception_AI", agent_reception)
workflow.add_node("Business_Logic", logic_fraud_check)

workflow.set_entry_point("Reception_AI")

def route_reception(state):
    if state["identified_amount"] > 0: return "Business_Logic"
    return END

workflow.add_conditional_edges("Reception_AI", route_reception)
workflow.add_edge("Business_Logic", END)

app_graph = workflow.compile()

# Visualize the architecture
try:
    display(Image(app_graph.get_graph().draw_mermaid_png()))
except:
    pass'''),

        new_markdown_cell(r'''### Play with the Graph Engine
See how it responds differently to "I broke 2 legs" vs "I need 400 euros" vs "I demand 5000 euros immediately".'''),
        new_code_cell(r'''run_test = app_graph.invoke({"user_input": "I demand 5000 euros immediately for my pain!", "identified_amount": 0})
print(f"\nExtracted Amount: {run_test['identified_amount']}€")
print(f"Final Status: {run_test.get('status', 'WAITING FOR USER AMOUNT')}")'''),

        new_markdown_cell(r'''🚀 **FINAL TAKEAWAY:** In enterprise integrations, LLMs bring elasticity (handling chaotic human input), but classical workflow graphs ensure safety, legal compliance, and deterministic execution.''')
    ])

    with open(r'd:\Python\notebooks\notebooks\tuto_llm_to_agents_V2_EN.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    create_workshop_notebook()
