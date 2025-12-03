
import streamlit as st
import os
from dotenv import load_dotenv
from analyzers.mock_analyzer import MockVLMAnalyzer
from analyzers.gemini_analyzer import GeminiProAnalyzer  # Import the new analyzer
from core.analyzers import AccidentAnalyzer
from core.schemas import AnalysisResult

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

# --- UI Layout ---
st.title("🚗 VLM Accident Analysis Prototype")
st.markdown("Upload a video to analyze potential insurance liabilities and risks.")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    options = list(ANALYZER_CATALOG.keys())

    selected_models = st.multiselect(
        "Select Model(s) for Comparison",
        options=options,
        default=[]
    )

    st.markdown("---")
    st.info(
        "Select one or more models to compare their analysis."
    )
    if any(model.startswith("Gemini") for model in selected_models) and not os.getenv("GEMINI_API_KEY"):
        st.warning("Please provide your Gemini API key in a `.env` file to use Gemini models.")


# --- Main Content ---
uploaded_file = st.file_uploader(
    "Upload a dashcam or CCTV video",
    type=["mp4", "mov", "avi"]
)

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    video_path = os.path.join("videos", uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Use an expander for a cleaner, on-demand video player
    with st.expander("🎬 Preview Video", expanded=False):
        _, video_col, _ = st.columns([0.4, 0.2, 0.4])
        with video_col:
            st.video(video_path)

    # --- Analysis Trigger ---
    if st.button("Analyze Video", type="primary"):
        if not selected_models:
            st.warning("Please select at least one model from the sidebar to begin analysis.")
        else:
            results = {}
            gemini_models = [m for m in selected_models if m.startswith("Gemini")]
            other_models = [m for m in selected_models if not m.startswith("Gemini")]

            uploaded_video_file = None

            with st.status("Running analysis...", expanded=True) as status:
                try:
                    # --- Gemini Analysis (with single upload) ---
                    if gemini_models:
                        st.write("Uploading video to Gemini... (This may take a moment)")
                        uploaded_video_file = GeminiProAnalyzer.upload_video(video_path)

                        for model_name in gemini_models:
                            try:
                                st.write(f"Analyzing with {model_name}...")
                                analyzer = get_analyzer(model_name)
                                results[model_name] = analyzer.perform_analysis_on_file(uploaded_video_file)
                            except Exception as e:
                                st.error(f"An error occurred with {model_name}: {e}")
                                results[model_name] = f"An error occurred: {e}"

                    # --- Other Models Analysis ---
                    for model_name in other_models:
                        if ANALYZER_CATALOG[model_name][0] is None:
                            results[model_name] = "Model not implemented."
                            continue
                        try:
                            st.write(f"Analyzing with {model_name}...")
                            analyzer = get_analyzer(model_name)
                            results[model_name] = analyzer.analyze_video(video_path)
                        except Exception as e:
                            st.error(f"An error occurred with {model_name}: {e}")
                            results[model_name] = f"An error occurred: {e}"

                finally:
                    # --- Cleanup ---
                    if uploaded_video_file:
                        st.write("Cleaning up uploaded resources...")
                        GeminiProAnalyzer.delete_video(uploaded_video_file)

                status.update(label="Analysis complete!", state="complete", expanded=False)

            # --- Results Display ---
            st.header("Analysis Comparison")

            # Reorder results to match selection order
            display_tabs = st.tabs([name for name in selected_models if name in results])

            for i, model_name in enumerate(selected_models):
                if model_name not in results:
                    continue

                with display_tabs[i]:
                    result = results[model_name]

                    if isinstance(result, AnalysisResult):
                        # --- Enhanced Dashboard View ---
                        st.subheader(f"Dashboard for {model_name}")
                        
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
                    else:
                        # Display error or info message
                        st.error(f"Could not generate a report for {model_name}.")
                        st.code(str(result), language=None)
