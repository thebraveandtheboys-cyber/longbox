# roughcut

First-pass editor for long talking-head videos. It does the slow part of the edit
and hands Premiere Pro a normal, fully editable timeline.

- **Syncs** a separate mic recording to the camera audio
- **Cuts** long pauses, filler words (um, uh) and false starts (last take wins)
- **Covers jump cuts with b-roll** on V2, matching clips to what's being said
- **Exports Premiere XML**: File > Import gives you a sequence with every cut in place

## Setup

```
pip install -r tools/roughcut/requirements.txt   # faster-whisper, numpy
# plus ffmpeg on the PATH
```

## Use

```
R=tools/roughcut/roughcut.py

python3 $R sync camera.mp4 mic.wav                 # optional: separate mic
python3 $R transcribe camera.mp4 --model medium    # word-timed transcript
python3 $R plan transcript.json --sync sync.json --broll ./broll
python3 $R report edl.json                         # read the cut list
python3 $R preview edl.json                        # quick low-res check
python3 $R xml edl.json --media-root "/Users/you/Footage/Shoot 12"
```

Then in Premiere: **File > Import > roughcut.xml**. If Premiere asks for media,
point it at your footage folder. Passing `--media-root` up front avoids that.

## B-roll library

Put clips in one folder. File names are the tags: `coffee-pour-closeup.mp4`
matches "coffee" and "pour". To add tags without renaming, add a `broll.json` file:

```json
{ "IMG_0412.mov": ["kettle", "steam", "kitchen"] }
```

Each jump cut gets a clip starting a moment before the cut. The best keyword match
wins, and repeats are avoided. Use `--matched-only` to leave cuts uncovered when
nothing matches.

## Tuning

| flag | default | effect |
|---|---|---|
| `--max-pause` | 0.5 | pauses longer than this (seconds) are closed up |
| `--pad` | 0.12 | breathing room kept around words |
| `--broll-length` | 2.4 | target b-roll duration |
| `--keep-fillers` / `--keep-retakes` | off | turn those cuts off |
| `transcribe --model` | small | `medium` or `large-v3` is more accurate but slower |

## Editing the plan

`edl.json` is the edit. To restore a cut, merge the two segments on either side
of it. To change b-roll, edit `file`, `src_in` or the record times. Then re-run
`xml`. You don't need to transcribe again. With Claude, just say "restore cut 4"
or "put the kettle shot at 2:10".

## Limits

- Retake detection compares how phrases start. A restart that changes its first
  words won't be caught, so check `report` before importing.
- Timelines use one camera angle. For multicam, sync the angles in Premiere and
  then use this for the cut list.
