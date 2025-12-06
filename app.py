import streamlit as st
import os
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv
from analyzers.mock_analyzer import MockVLMAnalyzer
from analyzers.gemini_analyzer import GeminiProAnalyzer
from analyzers.openrouter_analyzer import OpenRouterAnalyzer
from core.analyzers import AccidentAnalyzer
from core.schemas import AnalysisResult
from core.evaluation import evaluate_sync_metrics, evaluate_llm_judge_metrics_async
from core.exporter import create_pdf_report
from core.agent import app as agent_app

# Load environment variables
load_dotenv()

# --- Configuration ---
st.set_page_config(page_title="VLM Accident Analysis", page_icon="🚗", layout="wide")

# --- Custom CSS for sidebar buttons ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] .stButton>button {
        border: none;
        background-color: transparent;
        text-align: left !important;
        display: block !important;
        width: 100%;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background-color: #eee;
        color: #000;
    }
    .stDownloadButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
    }
    .ai-assistant-button button {
        background-color: #8A2BE2; /* BlueViolet */
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Analyzer Strategy Mapping ---
ANALYZER_CATALOG = {
    "Mock VLM (Demo)": (MockVLMAnalyzer, {}),
    "Gemini 3 Pro (Preview)": (GeminiProAnalyzer, {"model_name": "gemini-3-pro-preview"}),
    "Gemini 2.5 Pro": (GeminiProAnalyzer, {"model_name": "gemini-2.5-pro"}),
    "Nemotron Nano 12B 2 VL": (OpenRouterAnalyzer, {"model_name": "nvidia/nemotron-nano-12b-v2-vl:free"}),
}

# --- UI Rendering Functions ---

def display_accuracy_scorecard(sync_scores, judge_scores):
    """Renders the final, domain-driven scorecard."""
    st.markdown("##### 🎯 Accuracy Scorecard")

    # --- Domain & Objective Scores ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Environment (F1)",
        f"{sync_scores['environment_score']:.2f}",
        help="F1-Score for time_of_day, weather, road_conditions, and location_type."
    )
    col2.metric(
        "Human Factors",
        f"{sync_scores['human_factors_score']:.2f}",
        help="Weighted accuracy for injury_risk (60%), pedestrians_involved (30%), and potential_witnesses (10%)."
    )
    col3.metric(
        "Vehicle Score",
        f"{sync_scores['vehicle_score']:.2f}",
        help="Average Match Quality for vehicle identification and attributes."
    )
    col4.metric(
        "Liability Score",
        f"{sync_scores['liability_score']:.2f}",
        help="Weighted similarity for identifying the at-fault party, based on behavior (50%), color (25%), and type (25%)."
    )

    # --- LLM Judge Rating ---
    st.markdown("###### Intelligent Summary Rating (LLM as a Judge)")
    scol1, scol2, scol3 = st.columns(3)
    scol1.metric(
        "Summary Rating (1-100)",
        f"{judge_scores['summary_rating']}",
        help="LLM Judge's score for the semantic accuracy and completeness of the accident summary."
    )
    scol2.metric("Summary (BLEU)", f"{sync_scores['summary_bleu']:.2f}", help="Phrase overlap for the summary.")
    scol3.metric("Summary (METEOR)", f"{sync_scores['summary_meteor']:.2f}", help="Semantic similarity for the summary.")

    st.markdown("---")

def display_performance_metrics(performance, judge_performance):
    """Renders the combined performance metrics with a breakdown."""
    st.markdown("##### ⏱️ Performance")

    total_performance = performance.copy()
    if judge_performance:
        for key in total_performance:
            total_performance[key] += judge_performance.get(key, 0)

    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    p_col1.metric(
        "Total Latency (s)",
        f"{total_performance['latency']:.2f}",
        delta=f"{judge_performance['latency']:.2f}s (Evaluate)" if judge_performance else None,
        help="Main analysis latency + accuracy evaluation latency."
    )
    p_col2.metric(
        "Total Est. Cost ($)",
        f"{total_performance['estimated_cost']:.4f}",
        delta=f"{judge_performance['estimated_cost']:.4f}$ (Evaluate)" if judge_performance else None,
        help="Main analysis cost + accuracy evaluation cost."
    )
    p_col3.metric(
        "Total Input Tokens",
        f"{total_performance['input_tokens']:,}",
        delta=f"{judge_performance['input_tokens']:,} (Evaluate)" if judge_performance else None,
        help="Main analysis tokens + accuracy evaluation tokens."
    )
    p_col4.metric(
        "Total Output Tokens",
        f"{total_performance['output_tokens']:,}",
        delta=f"{judge_performance['output_tokens']:,} (Evaluate)" if judge_performance else None,
        help="Main analysis tokens + accuracy evaluation tokens."
    )
    st.markdown("---")

def display_analysis_dashboard(result: AnalysisResult):
    """Renders the main analysis result dashboard."""
    st.subheader(f"Dashboard")

    # --- Top Level Metrics ---
    with st.container(border=True):
        st.markdown("##### 🚩 Key Flags")
        col1, col2, col3 = st.columns(3)
        with col1:
            # Correctly access injury_risk from the human_factors object
            if "high" in result.human_factors.injury_risk.lower():
                st.error(f"**Injury Risk:** {result.human_factors.injury_risk}")
            elif "medium" in result.human_factors.injury_risk.lower():
                st.warning(f"**Injury Risk:** {result.human_factors.injury_risk}")
            else:
                st.info(f"**Injury Risk:** {result.human_factors.injury_risk}")
        with col2:
            # Format the new liability object into a readable string
            st.warning(f"**At-Fault Party:** Color: {result.liability_indicator.color}, Type: {result.liability_indicator.type}")
        with col3:
            st.info(f"**Collision Type:** {result.collision_type}")

    with st.container(border=True):
      st.markdown("##### 📋 Summary & Actions")
      st.markdown(f"**Summary:** {result.accident_summary}")
      st.markdown(f"**Liability Reasoning:** {result.liability_indicator.driver_major_behavior}")
      st.markdown(f"**Recommended Action:** {result.recommended_action}")

    # --- Detailed Sections ---
    with st.container(border=True):
      c1, c2 = st.columns(2)
      with c1:
          st.markdown("##### 🌎 Environmental Conditions")
          st.markdown(f"**- Time of Day:** {result.environmental_conditions.time_of_day}")
          st.markdown(f"**- Weather:** {result.environmental_conditions.weather}")
          st.markdown(f"**- Road Conditions:** {result.environmental_conditions.road_conditions}")
          st.markdown(f"**- Location Type:** {result.environmental_conditions.location_type}")
          st.markdown(f"**- Traffic Controls:** {', '.join(result.traffic_controls_present)}")

      with c2:
          st.markdown("##### 👨‍👩‍👧 Human Factors")
          st.markdown(f"**- Pedestrians Involved:** {result.human_factors.pedestrians_involved}")
          st.markdown(f"**- Potential Witnesses:** {result.human_factors.potential_witnesses}")
          st.markdown(f"**- Injury Risk:** {result.human_factors.injury_risk}")

    with st.container(border=True):
      st.markdown("##### 🚗 Vehicles Involved")
      for vehicle in result.vehicles_involved:
          st.markdown(f"- **{vehicle.color} {vehicle.type}:** {vehicle.damage_level} damage to the {vehicle.damage_direction}.")

      # --- Raw Data/Reasoning ---
      with st.expander("Show System Reasoning & Raw Data"):
          st.markdown("###### Reasoning Trace")
          for step in result.reasoning_trace:
              st.markdown(f"- {step}")
          st.markdown("###### Raw JSON Output")
          st.json(result.model_dump_json(indent=4))

    st.markdown("---")

# --- Backend Logic ---
def log_message(status, message):
    timestamp = datetime.now().strftime('%H:%M:%S')
    status.write(f"`[{timestamp}]` {message}")

def log_success(status, message):
    timestamp = datetime.now().strftime('%H:%M:%S')
    status.markdown(f"`[{timestamp}]` <span style='color:green;'>{message}</span>", unsafe_allow_html=True)

def get_analyzer(model_name: str) -> AccidentAnalyzer:
    """Factory function to get an analyzer instance."""
    analyzer_class, kwargs = ANALYZER_CATALOG.get(model_name, (None, {}))
    if analyzer_class:
        # Check for API key
        if issubclass(analyzer_class, GeminiProAnalyzer) and not os.getenv("GEMINI_API_KEY"):
            st.error("GEMINI_API_KEY not found. Please create a .env file with your key.")
            st.stop()
        if issubclass(analyzer_class, OpenRouterAnalyzer) and not os.getenv("OPENROUTER_API_KEY"):
            st.error("OPENROUTER_API_KEY not found. Please create a .env file with your key.")
            st.stop()
        return analyzer_class(**kwargs)
    raise ValueError("Invalid analyzer selected")

async def run_analysis_async(selected_models: list, video_path: str, status):
    """Runs the analysis for all selected models concurrently."""
    results = {}
    gemini_models = {m for m in selected_models if m.startswith("Gemini")}
    uploaded_video_file = None

    async def run_and_tag(model_name, coro):
        """Wrapper to run a coroutine and tag its result with the model name."""
        try:
            result = await coro
            return model_name, result, None
        except Exception as e:
            return model_name, None, e

    try:
        tasks = []
        if gemini_models:
            log_message(status, "Uploading video to Gemini...")
            uploaded_video_file = await GeminiProAnalyzer.upload_video(video_path)
        for model_name in selected_models:
            if ANALYZER_CATALOG[model_name][0] is None:
                results[model_name] = ("Model not implemented.", None)
                continue
            log_message(status, f"Queuing analysis for {model_name}...")
            analyzer = get_analyzer(model_name)
            coro = analyzer.analyze_video(video_path) if model_name not in gemini_models else analyzer.perform_analysis_on_file(uploaded_video_file, video_path)
            tasks.append(run_and_tag(model_name, coro))

        log_message(status, f"Running {len(tasks)} tasks in parallel...")
        for future in asyncio.as_completed(tasks):
            model_name, result_tuple, error = await future
            if error:
                st.error(f"An error occurred with {model_name}: {error}")
                results[model_name] = (None, f"An error occurred: {error}")
            else:
                log_success(status, f"Completed analysis for {model_name}.")
                results[model_name] = result_tuple
    finally:
        if uploaded_video_file:
            log_message(status, "Cleaning up uploaded resources...")
            await GeminiProAnalyzer.delete_video(uploaded_video_file)

    return results

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'selected_report' not in st.session_state:
    st.session_state.selected_report = "— New Analysis —"
if 'is_new_analysis' not in st.session_state:
    st.session_state.is_new_analysis = True

async def main():
    """The main async function for the Streamlit app."""
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.title("🚗 VLM Accident Analysis Prototype")

    # --- Sidebar ---
    with st.sidebar:
        if st.button("➕ New Analysis"):
            st.session_state.clear()
            st.rerun()

        st.markdown("---")
        st.header("Analysis")

        report_files = sorted([f for f in os.listdir("reports") if f.endswith(".json")], reverse=True)

        for report_file in report_files:
            display_name = report_file.replace(".json", "")
            if st.button(display_name, key=report_file, use_container_width=True):
                st.session_state.selected_report = report_file
                try:
                    with open(os.path.join("reports", report_file), "r") as f:
                        st.session_state.results = json.load(f)
                    st.session_state.is_new_analysis = False
                except Exception as e:
                    st.error(f"Failed to load report: {e}")
                    st.session_state.results = None
                st.rerun()

    # --- Main Content ---
    if st.session_state.get('results'):
        with col2:
            pdf_bytes = create_pdf_report(st.session_state.results)
            st.download_button(
                label="Export as PDF Report",
                data=pdf_bytes,
                file_name=f"{st.session_state.results['report_id']}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        # When displaying a result, the judge model is read from the saved config
        judge_model = st.session_state.results.get("config", {}).get("judge_model", "gemini-2.5-flash")
        await display_results_ui(st.session_state.results, judge_model)
    else:
        await run_new_analysis_ui()

async def run_new_analysis_ui():
    """The UI for running a new analysis."""
    st.markdown("Upload a video to analyze potential insurance liabilities and risks.")

    # --- Configuration for New Analysis ---
    st.markdown("##### Configuration")
    col1, col2 = st.columns(2)
    with col1:
        selected_models = st.multiselect("Select Model(s) for Comparison", options=list(ANALYZER_CATALOG.keys()), default=[])
    with col2:
        judge_model = st.selectbox("Select Judge Model (for accuracy)", options=["gemini-2.5-flash", "gemini-2.5-flash-lite"], index=0)

    if any(model.startswith("Gemini") for model in selected_models) and not os.getenv("GEMINI_API_KEY"):
        st.warning("Please provide your Gemini API key in a `.env` file to use Gemini models.")

    uploaded_file = st.file_uploader("Upload a dashcam or CCTV video", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        video_path = os.path.join("videos", uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        golden_data = None
        base_filename, _ = os.path.splitext(uploaded_file.name)
        golden_filename = f"{base_filename}.golden.json"
        golden_filepath = os.path.join("benchmark", "golden_files", golden_filename)
        if os.path.exists(golden_filepath):
            try:
                with open(golden_filepath, "r") as f:
                    golden_data = json.load(f)
                st.success(f"Found and loaded ground truth file: **{golden_filename}**")
            except Exception as e:
                st.warning(f"Found ground truth file, but failed to load it: {e}.")
                golden_data = None

        with st.expander("🎬 Preview Video", expanded=False):
            _, video_col, _ = st.columns([0.4, 0.2, 0.4])
            with video_col:
                st.video(video_path)

        if st.button("Analyze Video", type="primary", key="analyze_button"):
            if not selected_models:
                st.warning("Please select at least one model.")
            else:
                with st.status("Running analysis...", expanded=True) as status:
                    analysis_results = await run_analysis_async(selected_models, video_path, status)

                    # Construct the full results object
                    base_filename, _ = os.path.splitext(uploaded_file.name)
                    report_id = f"{base_filename}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                    results_obj = {
                        "report_id": report_id,
                        "video_filename": uploaded_file.name,
                        "config": {
                            "selected_models": selected_models,
                            "judge_model": judge_model
                        },
                        "models": {}
                    }

                    for model_name, (result, performance) in analysis_results.items():
                        # Run evaluation on the fly before saving
                        sync_scores = None
                        judge_scores = None
                        judge_performance = None
                        if golden_data and isinstance(result, AnalysisResult):
                            log_message(status, f"Evaluating accuracy for {model_name}...")
                            sync_scores = evaluate_sync_metrics(result, golden_data)
                            judge_scores, judge_performance = await evaluate_llm_judge_metrics_async(result, golden_data, judge_model)
                            log_success(status, f"Completed accuracy evaluation for {model_name}")

                        results_obj["models"][model_name] = {
                            "result": result.model_dump() if result else None,
                            "performance": performance,
                            "sync_scores": sync_scores,
                            "judge_scores": judge_scores,
                            "judge_performance": judge_performance
                        }

                    # Automatic Saving
                    report_filename = f"{results_obj['report_id']}.json"
                    with open(os.path.join("reports", report_filename), "w") as f:
                        json.dump(results_obj, f, indent=4)
                    st.toast(f"Analysis automatically saved as `{report_filename}`")

                    st.session_state.results = results_obj
                    st.session_state.is_new_analysis = False
                    status.update(label="All tasks complete!", state="complete")
                    st.rerun()

async def display_ai_assistant_ui(result, model_name):
    """Renders the UI for the AI Claims Assistant."""
    st.markdown('<div class="ai-assistant-button">', unsafe_allow_html=True)
    run_button = st.button("🤖 Run AI Claims Assistant", key=f"agent_button_{model_name}", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if run_button:
        st.markdown("##### 🤖 AI Claims Assistant Results")
        final_state = None
        with st.expander("Show Agent's Thought Process", expanded=True):
            # Prepare the initial message for the agent
            initial_prompt = f"""
            You are an AI Claims Assistant. You have just received a structured analysis of a car accident video.
            Your goal is to enrich this data using your available tools and provide a final, augmented recommendation and fraud risk assessment.

            **VLM Analysis:**
            {json.dumps(result.model_dump(), indent=2)}

            """
            initial_state = {"messages": [("user", initial_prompt)]}
            
            st.markdown("- **Running Too-Usering Agent...**")
            # Stream the graph execution
            for step in agent_app.stream(initial_state):
                node_name = list(step.keys())[0]
                # We don't need to show the full state, just the node name
                st.markdown(f"- **Running Node:** `{node_name}`")
                st.json(step[node_name], expanded=False)
                final_state = step[node_name]

        if final_state:
            st.markdown("###### Final Results")
            final_content = final_state["messages"][-1].content
            st.markdown(final_content)

            # --- Save Agent Results to Session State & Reports
            st.session_state.results["models"][model_name]["agent_results"] = final_content
            report_filename = f"{st.session_state.results['report_id']}.json"
            with open(os.path.join("reports", report_filename), "w") as f:
                json.dump(st.session_state.results, f, indent=4)
            st.toast("Agent results save to report.")

    st.markdown("---")


async def display_results_ui(results_data, judge_model):
    """Renders the results from a saved report or a new analysis."""
    st.header("Analysis Comparison")
    st.info(f"Displaying results for **{results_data['video_filename']}**")

    # --- Display Saved Configuration ---
    st.markdown("##### Configuration Used")
    col1, col2 = st.columns(2)
    with col1:
        st.multiselect("Models Used", options=results_data["config"]["selected_models"], default=results_data["config"]["selected_models"], disabled=True)
    with col2:
        st.text_input("Judge Model Used", value=results_data["config"]["judge_model"], disabled=True)

    display_tabs = st.tabs(list(results_data["models"].keys()))

    for i, model_name in enumerate(results_data["models"].keys()):
        with display_tabs[i]:
            model_data = results_data["models"][model_name]
            result_dict = model_data.get("result")
            performance = model_data.get("performance")

            if result_dict:
                result = AnalysisResult(**result_dict)

                # Get saved scores from the report
                sync_scores = model_data.get("sync_scores")
                judge_scores = model_data.get("judge_scores")
                judge_performance = model_data.get("judge_performance")

                # --- Display main results immediately ---
                display_analysis_dashboard(result)
                display_performance_metrics(performance, judge_performance)

                # --- Handle Accuracy Evaluation ---
                # If scores are already saved in the report, just display them
                if sync_scores and judge_scores:
                    display_accuracy_scorecard(sync_scores, judge_scores)
                # If this is a new analysis, run the evaluation now
                elif st.session_state.get('is_new_analysis', False):
                    golden_data = None
                    base_filename, _ = os.path.splitext(results_data['video_filename'])
                    golden_filename = f"{base_filename}.golden.json"
                    golden_filepath = os.path.join("benchmark", "golden_files", golden_filename)
                    if os.path.exists(golden_filepath):
                        with open(golden_filepath, "r") as f:
                            golden_data = json.load(f)

                    if golden_data:
                        with st.spinner(f"Evaluating accuracy for {model_name}..."):
                            sync_scores = evaluate_sync_metrics(result, golden_data)
                            judge_scores, judge_performance = await evaluate_llm_judge_metrics_async(result, golden_data, judge_model)

                            # Update the session state with the new scores for saving
                            st.session_state.results["models"][model_name]["sync_scores"] = sync_scores
                            st.session_state.results["models"][model_name]["judge_scores"] = judge_scores
                            st.session_state.results["models"][model_name]["judge_performance"] = judge_performance

                            # Rerun to update the UI with the new scores and performance
                            st.rerun()

                # Run assistant
                await display_ai_assistant_ui(result, model_name)
            else:
                st.error(f"Could not generate a report for {model_name}.")
                st.code(str(result_dict), language=None)
                st.code(str(performance), language=None)



if __name__ == "__main__":
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'selected_report' not in st.session_state:
        st.session_state.selected_report = "— New Analysis —"

    asyncio.run(main())
