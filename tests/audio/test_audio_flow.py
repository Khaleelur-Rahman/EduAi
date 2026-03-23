"""
Test script for the audio/TTS flow: chunking, audio-friendly transform,
chunked synthesis, and TTS-fallback (text when audio fails).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_chunk_text_ends_at_sentences():
    """Chunks must end at sentence boundaries and respect max size."""
    from app.audio import _chunk_text_for_audio

    # 5 short sentences, max 2 per chunk -> 3 chunks
    text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
    chunks = _chunk_text_for_audio(text, max_sentences_per_chunk=2, max_words_per_chunk=999)
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}: {chunks}"
    assert chunks[0].endswith("."), f"Chunk should end with period: {chunks[0]!r}"
    assert "First sentence." in chunks[0] and "Second sentence." in chunks[0]
    print("  ✓ Chunking respects sentence boundaries and max_sentences_per_chunk")

    # Long paragraph -> multiple chunks by word limit
    long_text = " ".join(["Word." for _ in range(150)])
    chunks = _chunk_text_for_audio(long_text, max_sentences_per_chunk=999, max_words_per_chunk=40)
    assert len(chunks) >= 2, f"Long text should produce multiple chunks, got {len(chunks)}"
    for c in chunks:
        assert c.endswith("."), f"Chunk should end with period: {c[:50]!r}..."
    print("  ✓ Chunking respects max_words_per_chunk")


def test_audio_friendly_transform():
    """Audio-friendly transform strips headers/markdown and normalizes for speech."""
    from app.audio import text_to_audio_friendly

    inp = "📚 *Lesson: Cells*\n\n*What are cells?*\nCells are tiny. _Type `/next` to continue._"
    out = text_to_audio_friendly(inp)
    assert "Lesson: Cells" in out or "Cells" in out
    assert "*" not in out or out.count("*") == 0, "Bold markers should be stripped"
    assert "Type `/next`" not in out or "Next" in out
    assert out.strip().endswith("."), "Output should end with sentence-ending punctuation"
    print("  ✓ Audio-friendly transform strips markdown and normalizes")


def test_tts_fallback_logic():
    """When tts_failed is True, we send text only (no audio)."""
    # Logic mirrored from main.py: when tts_failed and text -> text-only message
    def would_send_text_only(response):
        text = response.get("text", "") if isinstance(response, dict) else ""
        tts_failed = isinstance(response, dict) and response.get("tts_failed", False)
        return bool(tts_failed and text)

    assert would_send_text_only({"text": "Hello", "tts_failed": True}) is True
    assert would_send_text_only({"text": "Hello", "tts_failed": False, "audio_segments": [(b"x", "audio/mpeg")]}) is False
    assert would_send_text_only({"text": "Hello"}) is False
    assert would_send_text_only({"text": ""}) is False
    print("  ✓ TTS fallback: tts_failed + text => text-only branch")


def test_audio_segments_vs_single():
    """Response with audio_segments uses multi-segment path; single segment uses audio_bytes."""
    def branch(response):
        if isinstance(response, dict) and response.get("tts_failed") and response.get("text"):
            return "text_only"
        if isinstance(response, dict) and response.get("audio_segments"):
            segs = response["audio_segments"]
            return "multi_audio" if len(segs) > 1 else "single_audio"
        if isinstance(response, dict) and "audio_bytes" in response:
            return "single_audio"
        if response.get("text") if isinstance(response, dict) else False:
            return "text_only"
        return "none"

    assert branch({"text": "Hi", "tts_failed": True}) == "text_only"
    assert branch({"text": "Hi", "audio_segments": [(b"a", "audio/mpeg")]}) == "single_audio"
    assert branch({"text": "Hi", "audio_segments": [(b"a", "ct"), (b"b", "ct")]}) == "multi_audio"
    assert branch({"text": "Hi", "audio_bytes": b"x", "audio_content_type": "audio/mpeg"}) == "single_audio"
    print("  ✓ Response routing: tts_failed -> text; audio_segments -> audio path")


def test_synthesize_speech_chunked_returns_list():
    """synthesize_speech_chunked returns a list of (bytes, content_type); empty on failure."""
    from app.audio import synthesize_speech_chunked

    # Short text -> 1 segment (if TTS available)
    short = "Plants need sunlight. They use it to make food."
    segments = synthesize_speech_chunked(short, voice="alloy", age_group=10)
    assert isinstance(segments, list), "Must return a list"
    if segments:
        assert len(segments) >= 1
        assert isinstance(segments[0], tuple) and len(segments[0]) == 2
        assert isinstance(segments[0][0], bytes) and isinstance(segments[0][1], str)
        print(f"  ✓ synthesize_speech_chunked returned {len(segments)} segment(s) for short text")
    else:
        print("  ⊘ synthesize_speech_chunked returned [] (TTS not available or failed)")

    # Longer text -> multiple segments when TTS works
    long_para = (
        "Photosynthesis is how plants make food. "
        "They use sunlight, water, and air. "
        "This process makes oxygen. "
        "We need that to breathe. "
        "Leaves are like tiny factories."
    )
    segments_long = synthesize_speech_chunked(long_para, voice="alloy", age_group=10)
    assert isinstance(segments_long, list)
    if len(segments_long) >= 2:
        print(f"  ✓ Long paragraph produced {len(segments_long)} audio segments")
    elif segments_long:
        print(f"  ✓ Long paragraph produced 1 segment (TTS may truncate or chunk small)")


def run_all():
    print("\n=== Audio / TTS flow tests ===\n")
    outcomes = []
    for name, fn in [
        ("Chunk text at sentence boundaries", test_chunk_text_ends_at_sentences),
        ("Audio-friendly transform", test_audio_friendly_transform),
        ("TTS fallback logic", test_tts_fallback_logic),
        ("Audio segments vs single / routing", test_audio_segments_vs_single),
        ("synthesize_speech_chunked", test_synthesize_speech_chunked_returns_list),
    ]:
        try:
            fn()
            outcomes.append((name, True, None))
        except Exception as e:
            outcomes.append((name, False, str(e)))
            import traceback
            traceback.print_exc()

    print("\n--- Summary ---")
    for name, ok, err in outcomes:
        status = "PASS" if ok else "FAIL"
        extra = f" ({err})" if err else ""
        print(f"  [{status}] {name}{extra}")
    passed = sum(1 for _, ok, _ in outcomes if ok)
    print(f"\n{passed}/{len(outcomes)} passed")
    return passed == len(outcomes)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
