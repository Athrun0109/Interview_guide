import io
import tempfile
from dataclasses import dataclass, field

import torch
from pydub import AudioSegment

# Available Whisper models for transcription
WHISPER_MODELS = {
    "large-v3 (WhisperX)": "Large V3 + WhisperX (best quality, recommended for JA/EN)",
    "large-v3-turbo (WhisperX)": "Large V3 Turbo + WhisperX (faster, slightly lower quality)",
    "large-v3": "Large V3 (high quality, basic alignment)",
    "large-v3-turbo": "Large V3 Turbo (fast, basic alignment)",
    "medium": "Medium (faster, basic alignment)",
}

# Supported interview languages
SUPPORTED_LANGUAGES = {
    "ja": "Japanese (日本語)",
    "en": "English",
}

# Threshold for marking a segment as overlapping speech (fraction of duration
# that must overlap with a *second* speaker to flag it as OVERLAP).
OVERLAP_RATIO_THRESHOLD = 0.30


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


def _assign_speaker_with_overlap(
    start: float,
    end: float,
    diar_segments: list,
) -> str:
    """Pick the dominant speaker for [start, end]. Return 'OVERLAP' if a
    second speaker also covers >= OVERLAP_RATIO_THRESHOLD of the span."""
    duration = max(end - start, 1e-6)
    per_speaker: dict[str, float] = {}
    for ds in diar_segments:
        ov = max(0.0, min(end, ds["end"]) - max(start, ds["start"]))
        if ov > 0:
            per_speaker[ds["speaker"]] = per_speaker.get(ds["speaker"], 0.0) + ov

    if not per_speaker:
        return "UNKNOWN"

    ranked = sorted(per_speaker.items(), key=lambda kv: kv[1], reverse=True)
    top_spk, top_ov = ranked[0]
    if len(ranked) >= 2 and ranked[1][1] / duration >= OVERLAP_RATIO_THRESHOLD:
        return "OVERLAP"
    return top_spk


def _transcribe_with_whisperx(
    wav_path: str,
    hf_token: str,
    model_size: str,
    device: str,
    compute_type: str,
    language: str | None = None,
    num_speakers: int | None = None,
    progress_cb=None,
) -> TranscriptionResult:
    """Use WhisperX for transcription with word-level alignment and speaker diarization."""
    import whisperx
    import numpy as np

    def _p(label, frac):
        if progress_cb:
            progress_cb(label, frac)

    _p("Loading Whisper model...", 0.02)
    # Step 1: Load model and transcribe
    model = whisperx.load_model(model_size, device, compute_type=compute_type)

    _p("Loading audio...", 0.08)
    # Load audio using pydub (more reliable than whisperx.load_audio on Windows)
    audio_segment = AudioSegment.from_file(wav_path)
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / 32768.0
    audio = samples

    _p("Transcribing audio (this is the slowest step)...", 0.12)
    transcribe_kwargs = {"batch_size": 16}
    if language:
        transcribe_kwargs["language"] = language
    result = model.transcribe(audio, **transcribe_kwargs)
    detected_language = result.get("language", language or "en")

    _p("Aligning word-level timestamps...", 0.55)
    # Step 2: Align whisper output (word-level timestamps)
    align_model, align_metadata = whisperx.load_align_model(
        language_code=detected_language, device=device
    )
    result = whisperx.align(
        result["segments"], align_model, align_metadata, audio, device,
        return_char_alignments=False
    )

    _p("Identifying speakers (diarization)...", 0.70)
    # Step 3: Speaker diarization using pyannote directly
    from pyannote.audio import Pipeline as PyannotePipeline
    diarize_pipeline = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    diarize_pipeline.to(torch.device(device))
    diar_kwargs = {}
    if num_speakers is not None:
        diar_kwargs["num_speakers"] = num_speakers
    diarization = diarize_pipeline(wav_path, **diar_kwargs)

    _p("Aligning transcript with speakers...", 0.92)
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
            best_speaker = _assign_speaker_with_overlap(
                seg["start"], seg["end"], diar_segments
            )
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

                # Find speaker for this word (overlap-aware)
                best_speaker = _assign_speaker_with_overlap(
                    word_start, word_end, diar_segments
                )

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

    # Collect unique speakers (exclude UNKNOWN and OVERLAP from selectable list)
    speaker_labels = sorted(
        set(s["speaker"] for s in aligned_segments
            if s["speaker"] not in ("UNKNOWN", "OVERLAP"))
    )
    if not speaker_labels:
        speaker_labels = ["UNKNOWN"]

    speaker_samples = {}
    for spk in speaker_labels:
        sample = _extract_speaker_sample(wav_path, aligned_segments, spk)
        if sample:
            speaker_samples[spk] = sample

    _p("Done", 1.0)
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
    language: str | None = None,
    num_speakers: int | None = None,
    progress_cb=None,
) -> TranscriptionResult:
    """Use faster-whisper + pyannote for transcription (original method, segment-level alignment)."""
    from faster_whisper import WhisperModel
    from pyannote.audio import Pipeline

    def _p(label, frac):
        if progress_cb:
            progress_cb(label, frac)

    _p("Loading Whisper model...", 0.02)
    # Transcribe with faster-whisper
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Determine total audio duration so we can show real progress while streaming segments
    try:
        audio_duration = AudioSegment.from_file(wav_path).duration_seconds
    except Exception:
        audio_duration = 0.0

    fw_kwargs = {"word_timestamps": True}
    if language:
        fw_kwargs["language"] = language
    raw_segments, info = model.transcribe(wav_path, **fw_kwargs)

    # Stream segments — faster-whisper yields lazily, so this is where the slow work happens.
    # We can compute real progress as segment.end / total_duration.
    _p("Transcribing audio...", 0.05)
    whisper_segments = []
    for seg in raw_segments:
        whisper_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        })
        if audio_duration > 0:
            frac = 0.05 + 0.65 * min(seg.end / audio_duration, 1.0)
            _p(f"Transcribing audio... ({int(seg.end)}s / {int(audio_duration)}s)", frac)

    detected_language = info.language

    _p("Identifying speakers (diarization)...", 0.72)
    # Speaker diarization with pyannote
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    diarization_pipeline.to(torch.device(device))
    diar_kwargs = {}
    if num_speakers is not None:
        diar_kwargs["num_speakers"] = num_speakers
    diarization = diarization_pipeline(wav_path, **diar_kwargs)
    _p("Aligning transcript with speakers...", 0.92)

    # Build list of diarization segments
    diar_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diar_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    # Align whisper segments with diarization speakers (overlap-aware)
    aligned_segments = []
    for ws in whisper_segments:
        best_speaker = _assign_speaker_with_overlap(
            ws["start"], ws["end"], diar_segments
        )
        aligned_segments.append({
            "speaker": best_speaker,
            "start": ws["start"],
            "end": ws["end"],
            "text": ws["text"],
        })

    # Collect unique speakers (exclude UNKNOWN/OVERLAP from selectable list)
    speaker_labels = sorted(
        set(s["speaker"] for s in aligned_segments
            if s["speaker"] not in ("UNKNOWN", "OVERLAP"))
    )
    if not speaker_labels:
        speaker_labels = ["UNKNOWN"]
    speaker_samples = {}
    for spk in speaker_labels:
        sample = _extract_speaker_sample(wav_path, aligned_segments, spk)
        if sample:
            speaker_samples[spk] = sample

    _p("Done", 1.0)
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
    whisper_model_size: str = "large-v3 (WhisperX)",
    device: str = "cuda",
    compute_type: str = "float16",
    language: str | None = None,
    num_speakers: int | None = None,
    progress_cb=None,
) -> TranscriptionResult:
    """Run STT with speaker diarization.

    language: 'ja' or 'en' to force Whisper language; None for auto-detect.
    num_speakers: known speaker count (incl. candidate) to constrain diarization;
                  None falls back to pyannote auto-estimation.
    progress_cb: optional callable(label: str, fraction: float in [0,1]) for UI updates.
    """
    if progress_cb:
        progress_cb("Preparing audio...", 0.0)
    wav_path = _convert_to_wav(audio_bytes, file_name)
    use_whisperx = "(WhisperX)" in whisper_model_size

    if use_whisperx:
        model_name = whisper_model_size.replace(" (WhisperX)", "")
        return _transcribe_with_whisperx(
            wav_path, hf_token, model_name, device, compute_type,
            language=language, num_speakers=num_speakers,
            progress_cb=progress_cb,
        )
    else:
        return _transcribe_with_faster_whisper(
            wav_path, hf_token, whisper_model_size, device, compute_type,
            language=language, num_speakers=num_speakers,
            progress_cb=progress_cb,
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
