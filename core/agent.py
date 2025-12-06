# core/agent.py
import pandas as pd
import google.generativeai as genai
import asyncio
import json

# --- RAG Tools ---

def policy_lookup_tool(collision_type: str, at_fault_party: str) -> str:
    """
    Searches a dummy policy document to see if the incident is covered.
    In a real app, this would query a database or a vector store.
    """
    try:
        with open("knowledge_base/policy_123.txt", "r") as f:
            policy_text = f.read()
        
        # Simple string search for demonstration
        if collision_type.lower() in policy_text.lower() and "is covered" in policy_text.lower():
            return f"Finding: The policy document confirms that '{collision_type}' collisions are covered under the liability section."
        else:
            return f"Finding: Could not confirm coverage for '{collision_type}' in the policy document."
    except FileNotFoundError:
        return "Error: Policy document not found."

def claims_history_tool(at_fault_driver_name: str, collision_type: str) -> str:
    """
    Searches a dummy CSV to see if the at-fault driver has a history of similar claims.
    """
    try:
        history_df = pd.read_csv("knowledge_base/claims_history.csv")
        driver_claims = history_df[
            (history_df['driver_name'] == at_fault_driver_name) &
            (history_df['claim_type'] == collision_type) &
            (history_df['fault_status'] == 'At Fault')
        ]
        
        num_prior_claims = len(driver_claims)
        if num_prior_claims > 0:
            return f"Finding: Found {num_prior_claims} prior at-fault claim(s) of type '{collision_type}' for driver '{at_fault_driver_name}'."
        else:
            return f"Finding: No prior at-fault claims of type '{collision_type}' found for driver '{at_fault_driver_name}'."
    except FileNotFoundError:
        return "Error: Claims history file not found."

# --- Agentic Logic ---

async def run_claims_assistant_agent(analysis_result, agent_model_name: str):
    """
    Runs a simulated agentic workflow to enrich the VLM analysis.
    """
    thought_process = []
    
    agent_model = genai.GenerativeModel(agent_model_name)
    
    # --- Step 1: Initial thought based on VLM data ---
    initial_prompt = f"""
    You are an AI Claims Assistant. You have just received a structured analysis of a car accident video.
    Your goal is to enrich this data using your available tools and provide a final, augmented recommendation.

    **VLM Analysis:**
    - Collision Type: {analysis_result.collision_type}
    - At-Fault Party: {analysis_result.liability_indicator.color} {analysis_result.liability_indicator.type}
    - At-Fault Behavior: {analysis_result.liability_indicator.driver_major_behavior}

    **Available Tools:**
    - `policy_lookup_tool(collision_type, at_fault_party)`
    - `claims_history_tool(at_fault_driver_name, collision_type)`

    Based on the VLM analysis, what is the first tool you should use and what are the exact parameters?
    Respond in a JSON format like: {{\"tool_name\": \"tool_name\", \"parameters\": {{\"param1\": \"value1\"}}}}
    """
    
    thought_process.append("1. **Initial Thought:** Based on the VLM data, I need to check if the policy covers this type of incident.")
    
    # --- Step 2: Use the Policy Lookup Tool ---
    # (In a real agent, the LLM would choose this. Here we simulate the choice for simplicity.)
    tool_call_result = policy_lookup_tool(analysis_result.collision_type, "at_fault_party")
    thought_process.append(f"2. **Tool Call:** `policy_lookup_tool()`\n   **Result:** {tool_call_result}")

    # --- Step 3: Use the Claims History Tool ---
    # (Simulating the agent's next logical step)
    # For the demo, we'll assume the at-fault driver of the blue sedan is "John Doe"
    at_fault_driver = "John Doe" if "blue" in analysis_result.liability_indicator.color.lower() else "Jane Smith"
    
    history_call_result = claims_history_tool(at_fault_driver, analysis_result.collision_type)
    thought_process.append(f"3. **Tool Call:** `claims_history_tool(at_fault_driver_name='{at_fault_driver}')`\n   **Result:** {history_call_result}")

    # --- Step 4: Final Synthesis ---
    final_prompt = f"""
    You are an AI Claims Assistant. You have the following information:
    1.  **VLM Analysis:** Collision was a '{analysis_result.collision_type}'. The at-fault party was the '{analysis_result.liability_indicator.color} {analysis_result.liability_indicator.type}' due to '{analysis_result.liability_indicator.driver_major_behavior}'.
    2.  **Policy Check:** {tool_call_result}
    3.  **Claims History Check:** {history_call_result}

    Based on ALL of this information, generate a final "Augmented Recommendation" and a "Fraud Risk Assessment".
    The fraud risk should be "Low", "Medium", or "High" with a brief justification.
    Respond in a single JSON object: {{\"augmented_recommendation\": \"...\", \"fraud_risk_assessment\": \"...\", \"fraud_risk_justification\": \"...\"}}
    """
    
    thought_process.append("4. **Final Synthesis:** Combining all information to generate the final recommendation and fraud risk.")
    
    response = await asyncio.to_thread(agent_model.generate_content, final_prompt)
    try:
        json_part = response.text.split('```json')[1].split('```')[0]
        final_result = json.loads(json_part)
    except (IndexError, json.JSONDecodeError):
        final_result = {"augmented_recommendation": "Could not generate final result.", "fraud_risk_assessment": "Error"}

    return thought_process, final_result
