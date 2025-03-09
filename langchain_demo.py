[sophamie@rag agent]$ cat langchain_demo.py
#!/usr/bin/env python3
import os
from langchain.chat_models import ChatOpenAI
from langchain.agents import (
    initialize_agent,
    load_tools,
    AgentType,
    Tool
)

def poem_tool(topic: str) -> str:
    """Generate a multi-line poem about 'topic'."""
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    # We'll use ChatOpenAI here as well, for consistency
    poem_llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )
    # Just a direct prompt to produce a poem
    prompt = f"""Write a multi-line poem about "{topic}". 
Do not include lines like 'Thought:' or 'Action:' or 'Final Answer:' in your poem."""
    return poem_llm.predict(prompt).strip()

def main():
    # 1) Confirm OPENAI_API_KEY is set
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Please set OPENAI_API_KEY in your environment.")

    # 2) Create the main ChatOpenAI LLM
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo",
        temperature=0
    )

    # 3) Load the built-in "llm-math" tool, rename it to "Calculator"
    tools = load_tools(["llm-math"], llm=llm)
    tools[0].name = "Calculator"

    # 4) Create the custom "Poem" tool
    poem = Tool(
        name="Poem",
        func=poem_tool,
        description="Use this tool to write a multi-line poem about a given topic."
    )
    tools.append(poem)

    # 5) Initialize an agent using the recommended Chat ReAct type
    #
    #    We pass 'prefix' instructions that nudge the agent to use
    #    "Calculator" for math, "Poem" for poems, then combine results.
    #
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,  # Set to True so you can see how the agent thinks
        # If you get parsing errors, you can also add: handle_parsing_errors=True,
        agent_kwargs={
            "prefix": """You are a helpful AI using the ReAct framework.

You have two tools:
1) "Calculator" to do math.
2) "Poem" to write a multi-line poem.

When the user asks for math, call "Calculator".
When the user asks for a poem, call "Poem".
If the user asks for both, you must use both tools before you finalize the answer.

Combine all requested results into a single final answer at the end.
"""
        }
    )

    # 6) Our user query
    query = "What is 2025 * 45? Also, write a multi-line poem about the stars."
    print(f"\nUser Query:\n{query}\n")

    # 7) Run the agent
    response = agent.run(query)
    print("\nAgent's Final Response:\n", response)

if __name__ == "__main__":
    main()

[sophamie@rag agent]$ 

