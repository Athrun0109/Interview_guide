import re
import streamlit as st

from config import load_env_keys
from modules.transcriber import transcribe_and_diarize, format_transcript, WHISPER_MODELS
from modules.searcher import search_company
from modules.analyzer import analyze_interview, GEMINI_MODELS
from modules.prompts import AnalysisMode, determine_mode, build_exportable_prompt
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
    "job_title": "",
    "job_description": "",
    "company_name": "",
    "is_rejected": False,
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

    company_name = st.text_input("Company name (optional)", value=st.session_state.company_name)
    job_title = st.text_input("Job title", value=st.session_state.job_title)
    job_description = st.text_area("Job description / requirements", height=150, value=st.session_state.job_description)
    is_rejected = st.checkbox("I was rejected for this position", value=st.session_state.is_rejected)

    # Save to session state
    st.session_state.company_name = company_name
    st.session_state.job_title = job_title
    st.session_state.job_description = job_description
    st.session_state.is_rejected = is_rejected

    # ── Export Prompt Button ──────────────────────────────────────
    st.divider()
    st.subheader("Export for Other LLMs")
    st.caption("Copy this prompt to use with ChatGPT, Claude, or any other LLM.")

    if st.session_state.formatted_transcript and job_title:
        result = st.session_state.transcription_result
        exportable_prompt = build_exportable_prompt(
            transcript=st.session_state.formatted_transcript,
            job_title=job_title,
            job_description=job_description,
            company_name=company_name,
            is_rejected=is_rejected,
            detected_language=result.detected_language,
        )

        # Show prompt in a text area for easy copying
        with st.expander("View Full Prompt", expanded=False):
            st.text_area(
                "Complete prompt with all interview details",
                value=exportable_prompt,
                height=400,
                key="exportable_prompt_display",
            )
            st.info(f"Prompt length: {len(exportable_prompt):,} characters")
    else:
        st.warning("Please enter a job title to generate the exportable prompt.")

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
        elif not st.session_state.job_title:
            st.error("Please enter a job title.")
        else:
            # Determine mode
            search_summary = ""
            mode = determine_mode(st.session_state.is_rejected, st.session_state.company_name, search_summary)

            # Search company if applicable
            if st.session_state.company_name and mode != AnalysisMode.FAILURE:
                with st.spinner("Searching company background..."):
                    search_summary = search_company(st.session_state.company_name, st.session_state.job_title, serper_key)
                # Re-determine mode with search result
                mode = determine_mode(st.session_state.is_rejected, st.session_state.company_name, search_summary)

            # For FAILURE mode, still try to get company info if available
            if mode == AnalysisMode.FAILURE and st.session_state.company_name and serper_key:
                with st.spinner("Searching company background..."):
                    search_summary = search_company(st.session_state.company_name, st.session_state.job_title, serper_key)

            st.info(f"Analysis mode: **{mode.value}**")

            with st.spinner(f"Generating analysis with {gemini_model}..."):
                result = st.session_state.transcription_result
                report = analyze_interview(
                    transcript=st.session_state.formatted_transcript,
                    job_title=st.session_state.job_title,
                    job_description=st.session_state.job_description,
                    mode=mode,
                    api_key=gemini_key,
                    model=gemini_model,
                    detected_language=result.detected_language,
                    company_name=st.session_state.company_name,
                    search_summary=search_summary,
                )
                st.session_state.analysis_report = report

    # Display report with collapsible sections
    if st.session_state.analysis_report:
        st.divider()
        report = st.session_state.analysis_report

        # Parse the report into sections
        sections = {
            "overall": "",
            "qa": "",
            "focus": "",
            "suggestions": "",
            "went_well": "",
        }

        # Extract Overall Impression (always show)
        overall_match = re.search(
            r"###?\s*Overall Impression\s*\n(.*?)(?=\n###?\s*|\Z)",
            report,
            re.DOTALL | re.IGNORECASE
        )
        if overall_match:
            sections["overall"] = overall_match.group(1).strip()

        # Extract Q&A Breakdown
        qa_match = re.search(
            r"###?\s*Q&A Breakdown.*?\s*\n(.*?)(?=\n###?\s*|\Z)",
            report,
            re.DOTALL | re.IGNORECASE
        )
        if qa_match:
            sections["qa"] = qa_match.group(1).strip()

        # Extract Interviewer's Focus
        focus_match = re.search(
            r"###?\s*Interviewer'?s? Focus.*?\s*\n(.*?)(?=\n###?\s*|\Z)",
            report,
            re.DOTALL | re.IGNORECASE
        )
        if focus_match:
            sections["focus"] = focus_match.group(1).strip()

        # Extract Improvement Suggestions
        suggestions_match = re.search(
            r"###?\s*Improvement Suggestions?\s*\n(.*?)(?=\n###?\s*|\Z)",
            report,
            re.DOTALL | re.IGNORECASE
        )
        if suggestions_match:
            sections["suggestions"] = suggestions_match.group(1).strip()

        # Extract What Went Well (for FAILURE mode)
        went_well_match = re.search(
            r"###?\s*What Went Well\s*\n(.*?)(?=\n###?\s*|\Z)",
            report,
            re.DOTALL | re.IGNORECASE
        )
        if went_well_match:
            sections["went_well"] = went_well_match.group(1).strip()

        # Display sections
        st.subheader("Overall Impression")
        if sections["overall"]:
            st.markdown(sections["overall"])
        else:
            # Fallback: show first paragraph if parsing failed
            first_para = report.split("\n\n")[0] if "\n\n" in report else report[:500]
            st.markdown(first_para)

        # Collapsible detailed sections
        if sections["qa"]:
            with st.expander("Q&A Breakdown and Analysis", expanded=False):
                st.markdown(sections["qa"])

        if sections["went_well"]:
            with st.expander("What Went Well", expanded=False):
                st.markdown(sections["went_well"])

        if sections["focus"]:
            with st.expander("Interviewer's Focus Areas", expanded=False):
                st.markdown(sections["focus"])

        if sections["suggestions"]:
            with st.expander("Improvement Suggestions", expanded=False):
                st.markdown(sections["suggestions"])

        # Fallback: show full report if parsing failed
        if not any([sections["qa"], sections["focus"], sections["suggestions"]]):
            with st.expander("Full Analysis Report", expanded=True):
                st.markdown(report)
