import streamlit as st
import os
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv
from analyzers.mock_analyzer import MockVLMAnalyzer
from analyzers.gemini_analyzer import GeminiProAnalyzer
from core.analyzers import AccidentAnalyzer
from core.schemas import AnalysisResult
from core.evaluation import evaluate_sync_metrics, evaluate_llm_judge_metrics_async

# Load environment variables
load_dotenv()

# --- Configuration ---
st.set_page_config(
    page_title="VLM Accident Analysis",
    page_icon="🚗",
    layout="wide"
)

# --- Analyzer Strategy Mapping ---
ANALYZER_CATALOG = {
    "Mock VLM (Demo)": (MockVLMAnalyzer, {}),
    "Gemini 3 Pro (Preview)": (GeminiProAnalyzer, {"model_name": "gemini-3-pro-preview"}),
    "Gemini 2.5 Pro": (GeminiProAnalyzer, {"model_name": "gemini-2.5-pro"}),
}

# --- UI Rendering Functions ---

def display_accuracy_scorecard(sync_scores, judge_scores):
    """Renders the full, two-part accuracy scorecard."""
    st.markdown("##### 🎯 Accuracy Scorecard")

    # --- Part 1: Objective Metrics ---
    st.markdown("###### Objective Metrics (Word & Phrase Matching)")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Struct. F1", f"{sync_scores['categorical_f1']:.2f}", help="F1-Score for structured fields like weather, time of day, etc.")
    col2.metric("Injury Score", f"{sync_scores['injury_score']:.2f}", help="Hybrid score for injury classification and justification.")
    col3.metric("Summary (BLEU)", f"{sync_scores['summary_bleu']:.2f}", help="Phrase overlap for the summary.")
    col4.metric("Summary (METEOR)", f"{sync_scores['summary_meteor']:.2f}", help="Semantic similarity for the summary.")
    col5.metric("Liability (BLEU)", f"{sync_scores['liability_bleu']:.2f}", help="Phrase overlap for the liability statement.")
    col6.metric("Liability (METEOR)", f"{sync_scores['liability_meteor']:.2f}", help="Semantic similarity for the liability statement.")

    # --- Part 2: LLM Judge Ratings ---
    st.markdown("###### Intelligent Ratings (LLM as a Judge)")
    jcol1, jcol2, jcol3, jcol4, jcol5, jcol6 = st.columns(6)
    jcol1.metric("Summary (1-100)", f"{judge_scores['summary_rating']}", help="Judge's score for summary accuracy.")
    jcol2.metric("Liability (1-100)", f"{judge_scores['liability_rating']}", help="Judge's score for the liability assessment.")
    jcol3.metric("Damage Precision", f"{judge_scores['damage_precision']:.2f}", help="Judge's score for damage description relevance (Precision).")
    jcol4.metric("Damage Recall", f"{judge_scores['damage_recall']:.2f}", help="Judge's score for damage description completeness (Recall).")
    jcol5.metric("Behavior Precision", f"{judge_scores['behavior_precision']:.2f}", help="Judge's score for behavior flag relevance (Precision).")
    jcol6.metric("Behavior Recall", f"{judge_scores['behavior_recall']:.2f}", help="Judge's score for behavior flag completeness (Recall).")

    st.markdown("---")

def display_performance_metrics(performance, judge_performance):
    """Renders the combined performance metrics."""
    st.markdown("##### ⏱️ Performance")

    total_performance = performance.copy()
    if judge_performance:
        for key in total_performance:
            total_performance[key] += judge_performance[key]

    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    p_col1.metric(
        "Total Latency (s)",
        f"{total_performance['latency']:.2f}",
        help="Total time for analysis + accuracy evaluation."
    )
    p_col2.metric(
        "Total Est. Cost ($)",
        f"{total_performance['estimated_cost']:.4f}",
        help="Estimated cost for analysis + accuracy evaluation."
    )
    p_col3.metric(
        "Total Input Tokens",
        f"{total_performance['input_tokens']:,}",
        help="Input tokens for analysis + accuracy evaluation."
    )
    p_col4.metric(
        "Total Output Tokens",
        f"{total_performance['output_tokens']:,}",
        help="Output tokens for analysis + accuracy evaluation."
    )
    st.markdown("---")

def display_analysis_dashboard(result: AnalysisResult):
    """Renders the main analysis result dashboard."""
    st.subheader(f"Dashboard")

    # --- Top Level Metrics ---
    st.markdown("##### 🚩 Key Flags")
    col1, col2, col3 = st.columns(3)
    with col1:
        if "high" in result.injury_risk.lower():
            st.error(f"**Injury Risk:** {result.injury_risk}")
        elif "medium" in result.injury_risk.lower():
            st.warning(f"**Injury Risk:** {result.injury_risk}")
        else:
            st.info(f"**Injury Risk:** {result.injury_risk}")
    with col2:
        st.warning(f"**Liability:** {result.liability_indicator}")
    with col3:
        st.info(f"**Collision Type:** {result.collision_type}")

    st.markdown("---")

    # --- Detailed Sections ---
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🌎 Environmental Conditions")
        st.markdown(f"**- Time of Day:** {result.environmental_conditions.time_of_day}")
        st.markdown(f"**- Weather:** {result.environmental_conditions.weather}")
        st.markdown(f"**- Road Conditions:** {result.environmental_conditions.road_conditions}")
        st.markdown(f"**- Location Type:** {result.environmental_conditions.location_type}")

    with c2:
        st.markdown("##### 👨‍👩‍👧 Human Factors")
        st.markdown(f"**- Occupants Visible:** {result.human_factors.occupants_visible}")
        st.markdown(f"**- Pedestrians Involved:** {result.human_factors.pedestrians_involved}")
        st.markdown(f"**- Driver Behavior:** {', '.join(result.human_factors.driver_behavior_flags)}")
        st.markdown(f"**- Potential Witnesses:** {result.human_factors.potential_witnesses}")

    st.markdown("---")

    st.markdown("##### 📋 Summary & Actions")
    st.markdown(f"**Summary:** {result.accident_summary}")
    st.markdown(f"**Recommended Action:** {result.recommended_action}")

    st.markdown("##### 🚗 Vehicles Involved")
    for vehicle in result.vehicles_involved:
        st.markdown(f"- **Vehicle {vehicle.vehicle_id} ({vehicle.description}):** {vehicle.damage}")

    # --- Raw Data/Reasoning ---
    with st.expander("Show System Reasoning & Raw Data"):
        st.markdown("###### Reasoning Trace")
        for step in result.reasoning_trace:
            st.markdown(f"- {step}")
        st.markdown("###### Raw JSON Output")
        st.json(result.model_dump_json(indent=4))

# --- Backend Logic ---

def get_analyzer(model_name: str) -> AccidentAnalyzer:
    """Factory function to get an analyzer instance."""
    analyzer_class, kwargs = ANALYZER_CATALOG.get(model_name, (None, {}))
    if analyzer_class:
        # Check for API key if a Gemini model is selected
        if issubclass(analyzer_class, GeminiProAnalyzer) and not os.getenv("GEMINI_API_KEY"):
            st.error("GEMINI_API_KEY not found. Please create a .env file with your key.")
            st.stop()
        return analyzer_class(**kwargs)
    raise ValueError("Invalid analyzer selected")

async def run_analysis_async(selected_models: list, video_path: str, status):
    """Runs the analysis for all selected models concurrently and processes them as they complete."""
    results = {}
    gemini_models = {m for m in selected_models if m.startswith("Gemini")}
    uploaded_video_file = None

    def log_message(message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        status.write(f"`[{timestamp}]` {message}")

    def log_success(message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        status.markdown(f"`[{timestamp}]` <span style='color:green;'>{message}</span>", unsafe_allow_html=True)

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
            log_message("Uploading video to Gemini... (This may take a moment)")
            uploaded_video_file = await GeminiProAnalyzer.upload_video(video_path)
        for model_name in selected_models:
            if ANALYZER_CATALOG[model_name][0] is None:
                results[model_name] = ("Model not implemented.", None)
                continue
            log_message(f"Queuing analysis for {model_name}...")
            analyzer = get_analyzer(model_name)
            if model_name in gemini_models:
                coro = analyzer.perform_analysis_on_file(uploaded_video_file, video_path)
            else:
                coro = analyzer.analyze_video(video_path)
            tasks.append(run_and_tag(model_name, coro))

        log_message(f"Running {len(tasks)} tasks in parallel...")
        for future in asyncio.as_completed(tasks):
            model_name, result_tuple, error = await future
            if error:
                st.error(f"An error occurred with {model_name}: {error}")
                results[model_name] = (None, f"An error occurred: {error}")
            else:
                log_success(f"Completed analysis with {model_name}.")
                results[model_name] = result_tuple
    finally:
        if uploaded_video_file:
            log_message("Cleaning up uploaded resources...")
            await GeminiProAnalyzer.delete_video(uploaded_video_file)
    return results

async def main():
    """The main async function for the Streamlit app."""
    st.title("🚗 VLM Accident Analysis Prototype")
    st.markdown("Upload a video to analyze potential insurance liabilities and risks.")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")
        options = list(ANALYZER_CATALOG.keys())
        selected_models = st.multiselect("Select Model(s) for Comparison", options=options, default=[])
        st.markdown("---")
        st.header("Evaluation Settings")
        judge_model = st.selectbox("Select Judge Model (for accuracy)", options=["gemini-2.5-flash", "gemini-2.5-flash-lite"], index=0, help="The model used to score semantic similarity for list-based fields.")
        st.markdown("---")
        st.info("Select one or more models to compare their analysis.")
        if any(model.startswith("Gemini") for model in selected_models) and not os.getenv("GEMINI_API_KEY"):
            st.warning("Please provide your Gemini API key in a `.env` file to use Gemini models.")

    # --- Main Content ---
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
                st.warning(f"Found ground truth file, but failed to load it: {e}. Accuracy will not be evaluated.")
                golden_data = None

        with st.expander("🎬 Preview Video", expanded=False):
            _, video_col, _ = st.columns([0.4, 0.2, 0.4])
            with video_col:
                st.video(video_path)

        if st.button("Analyze Video", type="primary"):
            if not selected_models:
                st.warning("Please select at least one model from the sidebar to begin analysis.")
            else:
                with st.status("Running analysis...", expanded=True) as status:
                    results = await run_analysis_async(selected_models, video_path, status)

                st.header("Analysis Comparison")
                display_tabs = st.tabs([name for name in selected_models if name in results])

                for i, model_name in enumerate(selected_models):
                    if model_name not in results: continue
                    with display_tabs[i]:
                        result, performance = results[model_name]

                        if isinstance(result, AnalysisResult):
                            display_analysis_dashboard(result)

                            judge_scores = None
                            judge_performance = None

                            if golden_data:
                                # Run both evaluations
                                sync_scores = evaluate_sync_metrics(result, golden_data)
                                with st.spinner(f"Getting LLM Judge ratings for {model_name}..."):
                                    judge_scores, judge_performance = await evaluate_llm_judge_metrics_async(result, golden_data, judge_model)

                                display_accuracy_scorecard(sync_scores, judge_scores)

                            display_performance_metrics(performance, judge_performance)
                        else:
                            # Display error or info message
                            st.error(f"Could not generate a report for {model_name}.")
                            st.code(str(result), language=None)
                            st.code(str(performance), language=None)

if __name__ == "__main__":
    asyncio.run(main())
