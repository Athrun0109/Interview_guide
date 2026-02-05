import streamlit as st

from config import load_env_keys
from modules.transcriber import transcribe_and_diarize, format_transcript, WHISPER_MODELS
from modules.searcher import search_company
from modules.analyzer import analyze_interview, GEMINI_MODELS
from modules.prompts import AnalysisMode, determine_mode
import config

st.set_page_config(page_title="AI Interview Analyzer", layout="wide")
st.title("AI Interview Analyzer")

# ── Sidebar: API Keys ─────────────────────────────────────────────
env_keys = load_env_keys()

with st.sidebar:
    st.header("API Keys")
    gemini_key = st.text_input(
        "Gemini API Key",
        value=env_keys["gemini_api_key"],
        type="password",
    )
    serper_key = st.text_input(
        "Serper API Key (optional)",
        value=env_keys["serper_api_key"],
        type="password",
    )
    hf_token = st.text_input(
        "HuggingFace Token",
        value=env_keys["hf_token"],
        type="password",
    )

# ── Session state defaults ────────────────────────────────────────
for key, default in {
    "transcription_result": None,
    "speaker_map": None,
    "formatted_transcript": None,
    "analysis_report": None,
    "step": 1,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ═══════════════════════════════════════════════════════════════════
# Step 1: Upload audio
# ═══════════════════════════════════════════════════════════════════
st.header("Step 1: Upload Interview Recording")

uploaded_file = st.file_uploader(
    "Upload audio or video file",
    type=["mp4", "mp3", "wav", "m4a", "webm", "ogg"],
)

if uploaded_file:
    st.audio(uploaded_file)

    # Whisper model selection
    whisper_model = st.selectbox(
        "Whisper Model",
        options=list(WHISPER_MODELS.keys()),
        format_func=lambda x: WHISPER_MODELS[x],
        index=0,  # default to large-v3-turbo
    )

    if st.button("Start Transcription"):
        if not hf_token:
            st.error("HuggingFace Token is required for speaker diarization.")
        else:
            with st.spinner(f"Transcribing with {whisper_model} and identifying speakers..."):
                result = transcribe_and_diarize(
                    audio_bytes=uploaded_file.getvalue(),
                    file_name=uploaded_file.name,
                    hf_token=hf_token,
                    whisper_model_size=whisper_model,
                    device=config.WHISPER_DEVICE,
                    compute_type=config.WHISPER_COMPUTE_TYPE,
                )
                st.session_state.transcription_result = result
                st.session_state.speaker_map = None
                st.session_state.formatted_transcript = None
                st.session_state.analysis_report = None
                st.session_state.step = 2
                st.rerun()

# ═══════════════════════════════════════════════════════════════════
# Step 2: Speaker identification
# ═══════════════════════════════════════════════════════════════════
if st.session_state.transcription_result and st.session_state.step >= 2:
    st.header("Step 2: Identify Yourself")
    st.write("Listen to each speaker sample and click **\"This is me\"** next to your voice.")

    result = st.session_state.transcription_result
    cols = st.columns(len(result.speaker_labels))

    for i, spk in enumerate(result.speaker_labels):
        with cols[i]:
            st.subheader(spk)
            sample = result.speaker_samples.get(spk)
            if sample:
                st.audio(sample, format="audio/wav")
            else:
                st.caption("(no sample available)")
            if st.button("This is me", key=f"pick_{spk}"):
                speaker_map = {}
                for label in result.speaker_labels:
                    speaker_map[label] = "Candidate" if label == spk else "Interviewer"
                st.session_state.speaker_map = speaker_map
                st.session_state.formatted_transcript = format_transcript(
                    result.segments, speaker_map
                )
                st.session_state.step = 3
                st.rerun()

    # Show transcript preview if speaker already selected
    if st.session_state.formatted_transcript:
        with st.expander("Transcript preview", expanded=False):
            st.text(st.session_state.formatted_transcript)

# ═══════════════════════════════════════════════════════════════════
# Step 3: Job information
# ═══════════════════════════════════════════════════════════════════
if st.session_state.speaker_map and st.session_state.step >= 3:
    st.header("Step 3: Job Information")

    company_name = st.text_input("Company name (optional)")
    job_title = st.text_input("Job title")
    job_description = st.text_area("Job description / requirements", height=150)
    is_rejected = st.checkbox("I was rejected for this position")

    if st.session_state.step < 4:
        st.session_state.step = 4

# ═══════════════════════════════════════════════════════════════════
# Step 4: Generate analysis
# ═══════════════════════════════════════════════════════════════════
if st.session_state.step >= 4:
    st.header("Step 4: Generate Analysis")

    # Gemini model selection
    gemini_model = st.selectbox(
        "Gemini Model",
        options=list(GEMINI_MODELS.keys()),
        format_func=lambda x: GEMINI_MODELS[x],
        index=0,  # default to gemini-2.0-flash
    )

    if st.button("Analyze Interview"):
        if not gemini_key:
            st.error("Gemini API Key is required.")
        elif not job_title:
            st.error("Please enter a job title.")
        else:
            # Determine mode
            search_summary = ""
            mode = determine_mode(is_rejected, company_name, search_summary)

            # Search company if applicable
            if company_name and mode != AnalysisMode.FAILURE:
                with st.spinner("Searching company background..."):
                    search_summary = search_company(company_name, job_title, serper_key)
                # Re-determine mode with search result
                mode = determine_mode(is_rejected, company_name, search_summary)

            # For FAILURE mode, still try to get company info if available
            if mode == AnalysisMode.FAILURE and company_name and serper_key:
                with st.spinner("Searching company background..."):
                    search_summary = search_company(company_name, job_title, serper_key)

            st.info(f"Analysis mode: **{mode.value}**")

            with st.spinner(f"Generating analysis with {gemini_model}..."):
                result = st.session_state.transcription_result
                report = analyze_interview(
                    transcript=st.session_state.formatted_transcript,
                    job_title=job_title,
                    job_description=job_description,
                    mode=mode,
                    api_key=gemini_key,
                    model=gemini_model,
                    detected_language=result.detected_language,
                    company_name=company_name,
                    search_summary=search_summary,
                )
                st.session_state.analysis_report = report

    # Display report
    if st.session_state.analysis_report:
        st.divider()
        st.subheader("Analysis Report")
        st.markdown(st.session_state.analysis_report)
