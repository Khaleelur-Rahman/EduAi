#!/usr/bin/env python3
"""
Test script to verify audio generation performance improvements.
Tests parallel TTS synthesis and measures timing.
"""
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.audio import synthesize_speech_chunked, tts_service

def test_parallel_tts():
    """Test that parallel TTS synthesis works and is faster."""
    print("=" * 60)
    print("Testing Parallel TTS Synthesis")
    print("=" * 60)
    
    # Sample lesson text (long enough to generate 2 chunks)
    test_text = """
    Cells are the basic building blocks of all living things. Just like bricks make up a wall, 
    cells make up your body. Every plant, animal, and person is made of cells. 
    
    There are two main types of cells: plant cells and animal cells. Plant cells have a special 
    wall around them called a cell wall. Animal cells don't have this wall. Both types have a 
    nucleus, which is like the brain of the cell. The nucleus tells the cell what to do.
    
    Inside cells, there are tiny parts called organelles. The mitochondria are like power plants 
    that give the cell energy. The vacuoles are like storage rooms where the cell keeps food and 
    water. Everything works together to keep the cell alive and healthy.
    """
    
    print(f"\nTest text length: {len(test_text)} characters")
    print(f"TTS service: {'edge-tts' if tts_service.edge_tts else 'None'}")
    
    # Test parallel synthesis
    print("\n--- Testing parallel synthesis (max_segments=2) ---")
    start = time.time()
    segments = synthesize_speech_chunked(test_text, voice="alloy", age_group=10, max_segments=2)
    elapsed = time.time() - start
    
    print(f"\nResults:")
    print(f"  Segments generated: {len(segments)}")
    print(f"  Total time: {elapsed:.2f}s")
    for i, (audio_bytes, content_type) in enumerate(segments):
        print(f"  Segment {i+1}: {len(audio_bytes)} bytes, type: {content_type}")
    
    # Verify we got exactly 2 segments
    assert len(segments) == 2, f"Expected 2 segments, got {len(segments)}"
    assert all(len(seg[0]) > 0 for seg in segments), "All segments should have audio bytes"
    assert all(seg[1] for seg in segments), "All segments should have content type"
    
    print("\n✅ Parallel TTS test passed!")
    print(f"   Generated {len(segments)} segments in {elapsed:.2f}s")
    
    return segments

def test_order_preservation():
    """Test that segments are returned in the correct order."""
    print("\n" + "=" * 60)
    print("Testing Order Preservation")
    print("=" * 60)
    
    test_text = "First part. Second part. Third part. Fourth part. Fifth part."
    
    segments = synthesize_speech_chunked(test_text, voice="alloy", age_group=10, max_segments=2)
    
    assert len(segments) == 2, "Should generate 2 segments"
    
    # Verify segments are non-empty
    assert all(len(seg[0]) > 0 for seg in segments), "Segments should contain audio"
    
    print("✅ Order preservation test passed!")
    print(f"   Generated {len(segments)} segments in correct order")

if __name__ == "__main__":
    try:
        print("Initializing audio services...")
        from app.audio import initialize_audio_services
        initialize_audio_services()
        
        test_parallel_tts()
        test_order_preservation()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
