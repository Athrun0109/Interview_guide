import io
import tempfile
from dataclasses import dataclass, field

import torch
from pydub import AudioSegment

# Available Whisper models for transcription
WHISPER_MODELS = {
    "large-v3-turbo": "Large V3 Turbo (fast, recommended)",
    "large-v3": "Large V3 (best quality, slower)",
    "medium": "Medium (faster, good quality)",
    "small": "Small (fastest, lower quality)",
}


@dataclass
class TranscriptSegment:
    speaker: str
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    segments: list[TranscriptSegment]
    detected_language: str
    speaker_labels: list[str]
    speaker_samples: dict[str, bytes] = field(default_factory=dict)


def _convert_to_wav(audio_bytes: bytes, file_name: str) -> str:
    """Convert uploaded audio/video to 16kHz mono WAV. Returns path to temp WAV file."""
    suffix = "." + file_name.rsplit(".", 1)[-1] if "." in file_name else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name

    audio = AudioSegment.from_file(tmp_in_path)
    audio = audio.set_channels(1).set_frame_rate(16000)

    wav_path = tmp_in_path.rsplit(".", 1)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path


def _extract_speaker_sample(audio_path: str, segments: list, speaker: str, max_seconds: float = 8.0) -> bytes:
    """Extract the longest continuous segment for a speaker as a WAV sample (up to max_seconds)."""
    speaker_segs = [s for s in segments if s["speaker"] == speaker]
    if not speaker_segs:
        return b""

    # Find the longest segment
    longest = max(speaker_segs, key=lambda s: s["end"] - s["start"])
    start_ms = int(longest["start"] * 1000)
    end_ms = min(int(longest["end"] * 1000), start_ms + int(max_seconds * 1000))

    audio = AudioSegment.from_file(audio_path)
    clip = audio[start_ms:end_ms]

    buf = io.BytesIO()
    clip.export(buf, format="wav")
    return buf.getvalue()


def transcribe_and_diarize(
    audio_bytes: bytes,
    file_name: str,
    hf_token: str,
    whisper_model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
) -> TranscriptionResult:
    """Run STT with faster-whisper then speaker diarization with pyannote.

    GPU memory note: both models fit in 16GB VRAM concurrently.
    Avoid explicit `del model` — CTranslate2 destructor can crash on some CUDA versions.
    """
    # Step 1: Convert to WAV
    wav_path = _convert_to_wav(audio_bytes, file_name)

    # Step 2: Transcribe with faster-whisper
    from faster_whisper import WhisperModel

    model = WhisperModel(whisper_model_size, device=device, compute_type=compute_type)
    raw_segments, info = model.transcribe(wav_path, word_timestamps=True)

    whisper_segments = []
    for seg in raw_segments:
        whisper_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        })

    detected_language = info.language
    # Do NOT explicitly delete the whisper model — CTranslate2's C++ destructor
    # can crash the process on certain CUDA/driver combinations.

    # Step 3: Speaker diarization with pyannote
    from pyannote.audio import Pipeline

    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    diarization_pipeline.to(torch.device(device))
    diarization = diarization_pipeline(wav_path)

    # Build list of diarization segments
    diar_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diar_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    # Step 6: Align whisper segments with diarization speakers (max overlap)
    aligned_segments = []
    for ws in whisper_segments:
        best_speaker = "UNKNOWN"
        best_overlap = 0.0

        for ds in diar_segments:
            overlap_start = max(ws["start"], ds["start"])
            overlap_end = min(ws["end"], ds["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = ds["speaker"]

        aligned_segments.append({
            "speaker": best_speaker,
            "start": ws["start"],
            "end": ws["end"],
            "text": ws["text"],
        })

    # Step 7: Collect unique speakers and extract samples
    speaker_labels = sorted(set(s["speaker"] for s in aligned_segments))
    speaker_samples = {}
    for spk in speaker_labels:
        sample = _extract_speaker_sample(wav_path, aligned_segments, spk)
        if sample:
            speaker_samples[spk] = sample

    # Build result
    transcript_segments = [
        TranscriptSegment(
            speaker=s["speaker"],
            start=s["start"],
            end=s["end"],
            text=s["text"],
        )
        for s in aligned_segments
    ]

    return TranscriptionResult(
        segments=transcript_segments,
        detected_language=detected_language,
        speaker_labels=speaker_labels,
        speaker_samples=speaker_samples,
    )


def format_transcript(segments: list[TranscriptSegment], speaker_map: dict[str, str]) -> str:
    """Format transcript segments into readable text.

    speaker_map: e.g. {"SPEAKER_00": "Candidate", "SPEAKER_01": "Interviewer"}
    Output format: [00:01 - 00:15] Interviewer: ...
    """
    lines = []
    for seg in segments:
        label = speaker_map.get(seg.speaker, seg.speaker)
        start_m, start_s = divmod(int(seg.start), 60)
        end_m, end_s = divmod(int(seg.end), 60)
        lines.append(f"[{start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}] {label}: {seg.text}")
    return "\n".join(lines)
