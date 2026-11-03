---
name: roughcut
description: Rough-cut a long talking-head video for Premiere Pro. Syncs a separate mic, cuts pauses, fillers and retakes, places b-roll over jump cuts, and exports Premiere XML. Use when the user hands over talking-head footage, asks to sync audio to video, or wants an edit, cut list or b-roll pass for Premiere.
---

# Talking-head rough cut

The tool is `tools/roughcut/roughcut.py`. Its README is in the same folder. It needs
ffmpeg plus `pip install -r tools/roughcut/requirements.txt`.

## Workflow

1. **Ask for what's missing**: the camera file, a separate mic file (if any), the
   b-roll folder, and where the media lives on their editing machine (for `--media-root`).
2. **Sync** (only with a separate mic): `sync camera mic`. Confidence under ~8 means
   the match is shaky. Say so, and suggest Premiere's Synchronize > Audio as a check.
3. **Transcribe**: `transcribe camera --model medium`. Use `small` for quick tests.
4. **Plan**: `plan transcript.json [--sync sync.json] [--broll DIR]`.
5. **Review before exporting**: run `report edl.json` and read it. Look for cuts that
   remove meaning, such as a "retake" that is really a repeated point made on purpose.
   Fix those in `edl.json` by merging the segments around the cut. Apply the user's
   content notes ("cut the tangent about X", "keep the intro") the same way, using
   the transcript to find times.
6. **Check**: `preview edl.json`, then pull a few frames or re-transcribe the preview
   to confirm the edit reads cleanly.
7. **Export**: `xml edl.json --media-root "<their footage folder>"`. Tell them:
   File > Import the .xml in Premiere, and relink if asked.

## edl.json

- `segments`: source time ranges kept from the camera file, in order.
- `cuts`: what was removed between segments, with the reason and the words.
- `broll`: V2 clips in timeline (record) time, with `src_in` into the clip.
- `audio`: the mic file plus the offset (seconds into the camera file where the mic starts).

After hand edits, re-run `xml` and `preview` only. Transcription doesn't need repeating.
