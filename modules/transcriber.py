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
    # Step 2: Align whisper output (word-level timestamps).
    # Japanese fails here on transformers>=4.57 because WhisperX's default model
    # (jonatasgrosman/wav2vec2-large-xlsr-53-japanese, 2021 format) no longer loads.
    # Falling back to segment-level timestamps still produces a usable transcript —
    # the speaker-assignment branch at result["segments"][i]["words"] == [] handles it.
    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=detected_language, device=device
        )
        result = whisperx.align(
            result["segments"], align_model, align_metadata, audio, device,
            return_char_alignments=False
        )
    except Exception as e:
        print(f"[transcriber] Word-level alignment skipped ({e}). "
              f"Using segment-level timestamps instead.")

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
        # Capture word-level timestamps (enabled via word_timestamps=True) — used below
        # to split a Whisper segment across speakers when diarization detects a turn mid-segment.
        words = []
        if seg.words:
            for w in seg.words:
                words.append({
                    "start": w.start,
                    "end": w.end,
                    "word": w.word,
                })
        whisper_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "words": words,
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

    # Diagnostic: report pyannote's raw cluster output before our processing.
    # If num_speakers=3 was requested but only 2 cluster IDs appear here,
    # pyannote itself failed to separate them (voice similarity), not our code.
    _pyannote_clusters = {}
    for ds in diar_segments:
        _pyannote_clusters[ds["speaker"]] = _pyannote_clusters.get(ds["speaker"], 0.0) + (ds["end"] - ds["start"])
    print(f"[transcriber] pyannote raw clusters: {len(_pyannote_clusters)} speakers, "
          f"durations: {[(k, round(v, 1)) for k, v in sorted(_pyannote_clusters.items())]}")

    # Align whisper segments with diarization speakers, at WORD granularity.
    # For each segment, walk through its words, look up the speaker for each word's
    # time range, and group consecutive same-speaker words into output entries.
    # Joining with "" (not " ") preserves natural spacing: faster-whisper emits
    # " hello" for English (leading space included) and "あなた" for Japanese
    # (no separator), so "".join produces correct text in both languages.
    aligned_segments = []
    for ws in whisper_segments:
        words = ws["words"]
        if not words:
            # Defensive fallback: shouldn't happen since word_timestamps=True is set,
            # but if a segment has no word data, fall back to segment-level assignment.
            best_speaker = _assign_speaker_with_overlap(
                ws["start"], ws["end"], diar_segments
            )
            aligned_segments.append({
                "speaker": best_speaker,
                "start": ws["start"],
                "end": ws["end"],
                "text": ws["text"],
            })
            continue

        # Pass 1: raw speaker per word.
        raw_speakers = [
            _assign_speaker_with_overlap(w["start"], w["end"], diar_segments)
            for w in words
        ]

        # Pass 2: smooth UNKNOWN words.
        # Japanese kana are ~0.1s — much finer than pyannote's turn boundary
        # precision, so an isolated kana can land in a sub-second gap between
        # turns and come back as UNKNOWN. Replace each UNKNOWN with the nearest
        # non-UNKNOWN neighbor in the same Whisper segment (prefer backward,
        # fall back to forward). If the whole segment had no pyannote coverage,
        # leave it UNKNOWN — that case is genuinely ambiguous.
        smoothed_speakers = list(raw_speakers)
        for i, spk in enumerate(raw_speakers):
            if spk != "UNKNOWN":
                continue
            replacement = None
            for j in range(i - 1, -1, -1):
                if raw_speakers[j] != "UNKNOWN":
                    replacement = raw_speakers[j]
                    break
            if replacement is None:
                for j in range(i + 1, len(raw_speakers)):
                    if raw_speakers[j] != "UNKNOWN":
                        replacement = raw_speakers[j]
                        break
            if replacement is not None:
                smoothed_speakers[i] = replacement

        # Group consecutive same-speaker words into output entries.
        current_speaker = None
        current_words = []
        current_start = None

        for word, spk in zip(words, smoothed_speakers):
            if spk != current_speaker:
                if current_words and current_speaker is not None:
                    aligned_segments.append({
                        "speaker": current_speaker,
                        "start": current_start,
                        "end": current_words[-1]["end"],
                        "text": "".join(w["word"] for w in current_words).strip(),
                    })
                current_speaker = spk
                current_words = [word]
                current_start = word["start"]
            else:
                current_words.append(word)

        if current_words and current_speaker is not None:
            aligned_segments.append({
                "speaker": current_speaker,
                "start": current_start,
                "end": current_words[-1]["end"],
                "text": "".join(w["word"] for w in current_words).strip(),
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


# Sentence-terminating punctuation we never want to follow with another period.
_TERMINAL_PUNCT = set("。、？！．，?!.,")


def _is_single_token(text: str, language: str) -> bool:
    """A segment with just one token is almost always a diarization boundary glitch,
    not a real utterance. JA: 1 character. EN: 1 whitespace-delimited word."""
    text = text.strip()
    if not text:
        return False
    if language == "ja":
        return len(text) == 1
    if language == "en":
        return len(text.split()) == 1
    return False


def _join_paragraph(a: str, b: str, language: str, *, connect_naturally: bool) -> str:
    """Concatenate two same-speaker segment texts. Insert 「。」 / ". " between
    substantial segments to keep the merged paragraph readable; flow tiny
    fragments straight into their neighbor."""
    a = a.rstrip()
    b = b.lstrip()
    if not a:
        return b
    if not b:
        return a
    sep_space = " " if language == "en" else ""
    if a[-1] in _TERMINAL_PUNCT or connect_naturally:
        return a + sep_space + b
    return a + (". " if language == "en" else "。") + b


def postprocess_segments(
    segments: list[TranscriptSegment],
    language: str,
) -> list[TranscriptSegment]:
    """Clean up diarized output. Called by the UI *after* the user picks which
    speaker is themselves, not inside transcribe_and_diarize — that way the
    raw diarization output stays inspectable if this cleanup misbehaves.

    Three sequential passes:

    1. OVERLAP resolution.
       - Refined rule: if the segment immediately before an OVERLAP is a single
         token AND the segments at -2 and +1 share the same real speaker, the
         single-token misattribution and the OVERLAP both inherit that speaker.
         This is the「学習とか / Candidate も / OVERLAP 結 / 構何…」case.
       - Default: an OVERLAP inherits its nearest real-speaker neighbor
         (backward preferred, forward fallback).

    2. Single-token sandwich. A 1-token non-OVERLAP segment between two
       same-other-speaker neighbors is re-assigned to that speaker (handles
       the pyannote single-kana misattribution that the OVERLAP pass didn't
       cover).

    3. Merge consecutive same-speaker entries into one paragraph, inserting
       「。」 / ". " between substantial pieces and flowing tiny fragments
       through without punctuation."""
    if not segments:
        return segments

    segs = [
        {"speaker": s.speaker, "start": s.start, "end": s.end, "text": s.text}
        for s in segments
    ]

    # Pass 1: OVERLAP resolution.
    for i, s in enumerate(segs):
        if s["speaker"] != "OVERLAP":
            continue
        prev1 = segs[i - 1] if i >= 1 else None
        prev2 = segs[i - 2] if i >= 2 else None
        next1 = segs[i + 1] if i + 1 < len(segs) else None

        if (prev1 is not None and prev2 is not None and next1 is not None
                and prev2["speaker"] == next1["speaker"]
                and prev2["speaker"] not in ("OVERLAP", "UNKNOWN")
                and prev1["speaker"] != prev2["speaker"]
                and _is_single_token(prev1["text"], language)):
            prev1["speaker"] = prev2["speaker"]
            s["speaker"] = prev2["speaker"]
            continue

        inherited = None
        for j in range(i - 1, -1, -1):
            if segs[j]["speaker"] not in ("OVERLAP", "UNKNOWN"):
                inherited = segs[j]["speaker"]
                break
        if inherited is None:
            for j in range(i + 1, len(segs)):
                if segs[j]["speaker"] not in ("OVERLAP", "UNKNOWN"):
                    inherited = segs[j]["speaker"]
                    break
        if inherited is not None:
            s["speaker"] = inherited

    # Pass 2: single-token sandwich for non-OVERLAP.
    for i in range(1, len(segs) - 1):
        s = segs[i]
        if s["speaker"] in ("OVERLAP", "UNKNOWN"):
            continue
        prev1, next1 = segs[i - 1], segs[i + 1]
        if (prev1["speaker"] == next1["speaker"]
                and prev1["speaker"] not in ("OVERLAP", "UNKNOWN")
                and prev1["speaker"] != s["speaker"]
                and _is_single_token(s["text"], language)):
            s["speaker"] = prev1["speaker"]

    # Pass 3: merge consecutive same-speaker.
    merged: list[dict] = []
    last_was_short = False
    for s in segs:
        s_is_short = _is_single_token(s["text"], language)
        if merged and merged[-1]["speaker"] == s["speaker"]:
            merged[-1]["text"] = _join_paragraph(
                merged[-1]["text"], s["text"], language,
                connect_naturally=(last_was_short or s_is_short),
            )
            merged[-1]["end"] = s["end"]
        else:
            merged.append(dict(s))
        last_was_short = s_is_short

    return [
        TranscriptSegment(
            speaker=m["speaker"], start=m["start"], end=m["end"], text=m["text"]
        )
        for m in merged
    ]


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
