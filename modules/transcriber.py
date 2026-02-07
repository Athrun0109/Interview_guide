import io
import tempfile
from dataclasses import dataclass, field

import torch
from pydub import AudioSegment

# Available Whisper models for transcription
WHISPER_MODELS = {
    "large-v3-turbo (WhisperX)": "Large V3 Turbo + WhisperX (best alignment, recommended)",
    "large-v3 (WhisperX)": "Large V3 + WhisperX (best quality)",
    "large-v3-turbo": "Large V3 Turbo (fast, basic alignment)",
    "large-v3": "Large V3 (high quality, basic alignment)",
    "medium": "Medium (faster, basic alignment)",
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


def _transcribe_with_whisperx(
    wav_path: str,
    hf_token: str,
    model_size: str,
    device: str,
    compute_type: str,
) -> TranscriptionResult:
    """Use WhisperX for transcription with word-level alignment and speaker diarization."""
    import whisperx
    import numpy as np

    # Step 1: Load model and transcribe
    model = whisperx.load_model(model_size, device, compute_type=compute_type)

    # Load audio using pydub (more reliable than whisperx.load_audio on Windows)
    audio_segment = AudioSegment.from_file(wav_path)
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / 32768.0
    audio = samples

    result = model.transcribe(audio, batch_size=16)
    detected_language = result.get("language", "en")

    # Step 2: Align whisper output (word-level timestamps)
    align_model, align_metadata = whisperx.load_align_model(
        language_code=detected_language, device=device
    )
    result = whisperx.align(
        result["segments"], align_model, align_metadata, audio, device,
        return_char_alignments=False
    )

    # Step 3: Speaker diarization using pyannote directly
    from pyannote.audio import Pipeline as PyannotePipeline
    diarize_pipeline = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    diarize_pipeline.to(torch.device(device))
    diarization = diarize_pipeline(wav_path)

    # Build diarization segments list
    diar_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diar_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    # Step 4: Assign speakers to each word, then rebuild segments
    # WhisperX provides word-level timestamps in result["segments"][i]["words"]
    aligned_segments = []
    for seg in result["segments"]:
        words = seg.get("words", [])
        if not words:
            # Fallback: no word-level data, use segment-level alignment
            best_speaker = "UNKNOWN"
            best_overlap = 0.0
            for ds in diar_segments:
                overlap_start = max(seg["start"], ds["start"])
                overlap_end = min(seg["end"], ds["end"])
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = ds["speaker"]
            aligned_segments.append({
                "speaker": best_speaker,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            })
        else:
            # Word-level alignment: group consecutive words by speaker
            current_speaker = None
            current_words = []
            current_start = None

            for word in words:
                word_start = word.get("start", seg["start"])
                word_end = word.get("end", seg["end"])
                word_text = word.get("word", "")

                # Find speaker for this word
                best_speaker = "UNKNOWN"
                best_overlap = 0.0
                for ds in diar_segments:
                    overlap_start = max(word_start, ds["start"])
                    overlap_end = min(word_end, ds["end"])
                    overlap = max(0.0, overlap_end - overlap_start)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_speaker = ds["speaker"]

                # Group words by speaker
                if best_speaker != current_speaker:
                    # Save previous group
                    if current_words and current_speaker is not None:
                        aligned_segments.append({
                            "speaker": current_speaker,
                            "start": current_start,
                            "end": current_words[-1].get("end", seg["end"]),
                            "text": " ".join(w.get("word", "") for w in current_words).strip(),
                        })
                    # Start new group
                    current_speaker = best_speaker
                    current_words = [word]
                    current_start = word_start
                else:
                    current_words.append(word)

            # Save last group
            if current_words and current_speaker is not None:
                aligned_segments.append({
                    "speaker": current_speaker,
                    "start": current_start,
                    "end": current_words[-1].get("end", seg["end"]),
                    "text": " ".join(w.get("word", "") for w in current_words).strip(),
                })

    # Collect unique speakers and extract samples
    speaker_labels = sorted(set(s["speaker"] for s in aligned_segments if s["speaker"] != "UNKNOWN"))
    if not speaker_labels:
        speaker_labels = ["UNKNOWN"]

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


def _transcribe_with_faster_whisper(
    wav_path: str,
    hf_token: str,
    model_size: str,
    device: str,
    compute_type: str,
) -> TranscriptionResult:
    """Use faster-whisper + pyannote for transcription (original method, segment-level alignment)."""
    from faster_whisper import WhisperModel
    from pyannote.audio import Pipeline

    # Transcribe with faster-whisper
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    raw_segments, info = model.transcribe(wav_path, word_timestamps=True)

    whisper_segments = []
    for seg in raw_segments:
        whisper_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        })

    detected_language = info.language

    # Speaker diarization with pyannote
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

    # Align whisper segments with diarization speakers (max overlap)
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

    # Collect unique speakers and extract samples
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


def transcribe_and_diarize(
    audio_bytes: bytes,
    file_name: str,
    hf_token: str,
    whisper_model_size: str = "large-v3-turbo (WhisperX)",
    device: str = "cuda",
    compute_type: str = "float16",
) -> TranscriptionResult:
    """Run STT with speaker diarization.

    If model name contains '(WhisperX)', use whisperX for better word-level alignment.
    Otherwise, use faster-whisper + pyannote with segment-level alignment.
    """
    # Convert to WAV
    wav_path = _convert_to_wav(audio_bytes, file_name)

    # Determine which method to use
    use_whisperx = "(WhisperX)" in whisper_model_size

    if use_whisperx:
        # Extract actual model name (e.g., "large-v3-turbo (WhisperX)" -> "large-v3-turbo")
        model_name = whisper_model_size.replace(" (WhisperX)", "")
        return _transcribe_with_whisperx(wav_path, hf_token, model_name, device, compute_type)
    else:
        return _transcribe_with_faster_whisper(wav_path, hf_token, whisper_model_size, device, compute_type)


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
