#!/usr/bin/env python3
"""
Generate test videos for the hybrid pipeline in one or more languages.
Saves videos to the project root as test_video_{lang}_{topic_slug}.mp4.

Usage:
  # Generate default set (en + es, 2 topics)
  python test_video_generate.py

  # Single language and topic
  python test_video_generate.py --lang es --topic "las células"

  # All supported languages, one topic
  python test_video_generate.py --topic photosynthesis --all-langs

  # Dry run (no actual generation)
  python test_video_generate.py --dry-run
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load .env before importing app
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ.setdefault("DATABASE_URL", "sqlite:///test.db")


def slug(s: str) -> str:
    """Safe filename slug from topic string."""
    s = re.sub(r"[^\w\s-]", "", s.lower())
    return re.sub(r"[-\s]+", "_", s).strip("_") or "topic"


def main():
    parser = argparse.ArgumentParser(description="Generate test videos (multi-language)")
    parser.add_argument("--topic", type=str, help="Single topic to use (e.g. photosynthesis)")
    parser.add_argument("--lang", type=str, default=None, help="Single language code (e.g. en, es)")
    parser.add_argument("--all-langs", action="store_true", help="Run for all supported languages")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be generated")
    args = parser.parse_args()

    from app.video import generate_lesson_video
    from app.language import SUPPORTED_LANGUAGES

    if args.all_langs:
        languages = list(SUPPORTED_LANGUAGES.keys())
    elif args.lang:
        lang = args.lang.strip().lower()
        if lang not in SUPPORTED_LANGUAGES:
            print(f"Unsupported language: {args.lang}. Supported: {list(SUPPORTED_LANGUAGES.keys())}")
            sys.exit(1)
        languages = [lang]
    else:
        languages = ["en", "es"]

    if args.topic:
        topics = [args.topic.strip()]
    else:
        topics = ["photosynthesis", "cells"]

    print("Video generation test")
    print("=" * 60)
    print(f"Topics: {topics}")
    print(f"Languages: {languages}")
    if args.dry_run:
        print("(dry run — no files will be written)")
    print()

    for topic in topics:
        for lang in languages:
            out_name = f"test_video_{lang}_{slug(topic)}.mp4"
            if args.dry_run:
                print(f"  [dry] {out_name} <- topic={topic!r}, lang={lang}")
                continue
            print(f"  Generating {out_name} ... ", end="", flush=True)
            start = time.time()
            try:
                out = generate_lesson_video(topic, language=lang, age_group=10)
                elapsed = time.time() - start
                if out:
                    video_bytes, content_type = out
                    Path(out_name).write_bytes(video_bytes)
                    print(f"OK ({len(video_bytes):,} bytes, {elapsed:.1f}s)")
                else:
                    print(f"FAILED (no output, {elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.time() - start
                print(f"ERROR ({elapsed:.1f}s): {e}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
