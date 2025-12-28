# core/agent.py
import pandas as pd
import json
import os
import sqlite3
from typing import TypedDict, Annotated, List
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# --- Agent initial state Definitions ---

def get_initial_state(result: str, driver_name = "John Doe") -> AgentState:
    # Prepare the initial message for the agent
    initial_prompt = f"""
    You are an AI Claims Assistant. You have just received a structured analysis of a car accident video.
    Your goal is to enrich this data using your available tools and provide a final, augmented recommendation and fraud risk assessment.

    ** Critical Instuctions::** 
    - Make sure verify policy coverage first before checking claim history.

    **Driver Name:** {driver_name}

    **VLM Analysis:**
    {result}

    """
    return {"messages": [("user", initial_prompt)]}

# --- Tool Definitions ---

@tool
def policy_lookup_tool(collision_type: str) -> str:
    """
    Searches the policy document to determine if a specific collision_type is covered.
    """
    try:
        with open("knowledge_base/policy_123.txt", "r") as f:
            policy_text = f.read()
        if collision_type.lower() in policy_text.lower() and "is covered" in policy_text.lower():
            return f"Finding: The policy document confirms that '{collision_type}' collisions are covered."
        else:
            return f"Finding: Could not confirm coverage for '{collision_type}' in the policy document."
    except FileNotFoundError:
        return "Error: Policy document not found."

@tool
def claims_history_tool(at_fault_driver_name: str, collision_type: str, vehicle_vin: str = None) -> str:
    """
    Queries the internal claims SQL database to find prior at-fault claims.
    Args:
        at_fault_driver_name: Name of the driver.
        collision_type: Type of collision (e.g., 'Rear-End').
        vehicle_vin: (Optional) The last few digits of the VIN if known.
    """
    db_path = "knowledge_base/claims_history.db"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Base Query
        query = """
            SELECT count(*), group_concat(claim_date)
            FROM claims
            WHERE driver_name = ?
            AND claim_type = ?
            AND fault_status = 'At Fault'
        """
        params = [at_fault_driver_name, collision_type]

        # Dynamic SQL: Add VIN filter if provided
        if vehicle_vin:
            query += " AND vehicle_vin LIKE ?"
            params.append(f"%{vehicle_vin}") # Matches '...456' even if partial

        cursor.execute(query, tuple(params))
        result = cursor.fetchone()
        conn.close()

        count = result[0]
        dates = result[1]

        if count > 0:
            msg = f"Finding: Found {count} prior at-fault claim(s) of type '{collision_type}' for driver '{at_fault_driver_name}'"
            if vehicle_vin:
                msg += f" in vehicle (VIN: {vehicle_vin})"
            return msg + f" on dates: {dates}."
        else:
            return f"Finding: No prior at-fault claims of type '{collision_type}' found for driver '{at_fault_driver_name}'."

    except Exception as e:
        return f"Database Error: {e}"

# --- Agent State Definition (Simplified) ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

# --- LLM and Tool Definitions ---
tools = [policy_lookup_tool, claims_history_tool]
tool_map = {tool.name : tool for tool in tools}
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.environ["GEMINI_API_KEY"], temperature=0)
llm_with_tools = llm.bind_tools(tools)

# --- Graph Nodes ---

def agent_node(state: AgentState):
    """The node that uses the LLM to decide the next step or to synthesize the final answer."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: AgentState):
    """This runs the tools and returns the results."""
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []
    for tool_call in tool_calls:
        tool_output = tool_map[tool_call["name"]].invoke(tool_call["args"])
        tool_messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
    return {"messages": tool_messages}

# --- Conditional Edges & Graph Definition ---

def router(state: AgentState):
    """The router function that directs the graph flow."""
    if state["messages"][-1].tool_calls:
        return "tool_node"
    else:
        return "END"

# --- Build the Graph ---
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tool_node", tool_node)

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    router,
    {"tool_node": "tool_node", "END": END}
)
graph.add_edge("tool_node", "agent")

# Compile the graph into a runnable app
app = graph.compile()
