# VLM Accident Analysis Prototype 🚗

This repository contains a **Streamlit-based application** for analyzing vehicle accident videos using advanced **Vision-Language Models (VLMs)**. The tool provides insights into accident scenarios, evaluates liability, and generates detailed reports.

---

## Features

- **Video Analysis**: Upload dashcam or CCTV footage for analysis.
- **Model Comparison**: Compare results from multiple VLMs, including Gemini, Qwen, and GPT models.
- **Accuracy Evaluation**: Evaluate model performance using ground truth data.
- **Interactive Dashboard**: Visualize analysis results, including environmental conditions, human factors, and vehicle details.
- **PDF Export**: Generate and download detailed PDF reports.
- **AI Claims Assistant**: Augment analysis with an AI-powered assistant for fraud detection and recommendations.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Helena6789/video-analysis.git
   cd video-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add API keys for the models (e.g., `GEMINI_API_KEY`, `OPENROUTER_API_KEY`).

---

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Open the app in your browser at `http://localhost:8501`.

3. Upload a video, select models, and start the analysis.

---

## Project Structure

- **`app.py`**: Main Streamlit application.
- **`core/`**: Core logic for analyzers, evaluation, and exporting reports.
- **`analyzers/`**: Implementations of various VLM analyzers.
- **`utils/`**: Utility functions for video processing and API interactions.
- **`benchmark/`**: Ground truth data for model evaluation.
- **`reports/`**: Generated analysis reports.
- **`templates/`**: HTML templates for report generation.

---

## Supported Models

- **Gemini 3 Pro (Preview)**
- **Gemini 2.5 Pro**
- **Nemotron Nano 12B 2 VL (Free)**
- **Qwen3 VL 235B A22B Thinking**
- **GPT-5.1**
- **OpenRouter Models**


---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- **Streamlit** for the interactive UI.
- **OpenAI, Google, NVIDIA** for the VLM models.