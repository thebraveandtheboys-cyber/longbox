#!/usr/bin/env python3
"""roughcut: first-pass editor for talking-head videos, built for Premiere Pro.

Steps (each writes a file the next one reads, so any step can be re-run or hand-edited):

  sync        line up a separate mic recording with the camera audio -> sync.json
  transcribe  word-timed transcript with faster-whisper            -> transcript.json
  plan        pick cuts (pauses, fillers, retakes) and b-roll       -> edl.json
  xml         Premiere-importable timeline (FCP7 XML)               -> roughcut.xml
  preview     quick low-res render to check the edit                -> preview.mp4
  report      readable list of every cut and b-roll placement       -> stdout

edl.json is the edit itself. Change it (restore a cut, swap a b-roll clip) and
re-run `xml` / `preview`; nothing is re-transcribed.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote
from xml.sax.saxutils import escape

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mxf", ".avi", ".mkv", ".webm"}
FILLERS = {"um", "umm", "uh", "uhh", "uhm", "erm", "er", "ah", "ahh", "hmm", "mm", "mhm"}
# Nudges Whisper to write fillers down instead of silently dropping them.
FILLER_PROMPT = "Umm, so, uh, I was like, hmm... okay. Uh, let me, um, start again."


# ---------------------------------------------------------------- helpers

def die(msg):
    sys.exit(f"roughcut: {msg}")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {path}")


def probe(path):
    """Duration, fps, size and audio layout of a media file."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True)
    if out.returncode:
        die(f"ffprobe failed on {path}: {out.stderr.strip()}")
    info = json.loads(out.stdout)
    res = {"path": str(Path(path).resolve()), "duration": float(info["format"].get("duration", 0)),
           "fps": None, "width": None, "height": None, "channels": 0, "samplerate": 48000}
    for s in info["streams"]:
        if s["codec_type"] == "video" and res["fps"] is None:
            num, den = s.get("avg_frame_rate", "0/1").split("/")
            if float(den) and float(num):
                res["fps"] = float(num) / float(den)
            res["width"], res["height"] = s.get("width"), s.get("height")
        elif s["codec_type"] == "audio" and not res["channels"]:
            res["channels"] = int(s.get("channels", 2))
            res["samplerate"] = int(s.get("sample_rate", 48000))
    return res


def norm(word):
    return re.sub(r"[^a-z0-9']", "", word.lower())


def stem(word):
    w = norm(word).replace("'", "")
    for suffix in ("ing", "ers", "er", "es", "ed", "s"):
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def ts(sec):
    m, s = divmod(max(sec, 0), 60)
    return f"{int(m)}:{s:05.2f}"


# ---------------------------------------------------------------- sync

def read_mono(path, sr, seconds):
    import numpy as np
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-t", str(seconds),
           "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32)


def find_offset(camera, mic, seconds=600, sr=8000):
    """Seconds into the camera file at which the mic recording starts (can be negative).

    Cross-correlates the two loudness envelopes; envelopes match well even when
    one mic is far away and echoey and the other is a close lav.
    """
    import numpy as np
    a, b = read_mono(camera, sr, seconds), read_mono(mic, sr, seconds)
    if not len(a) or not len(b):
        die("one of the files has no audio")

    def envelope(x):
        hop = sr // 200                     # 5 ms frames
        x = np.abs(x[: len(x) // hop * hop]).reshape(-1, hop).mean(axis=1)
        x = np.log1p(x)
        return x - x.mean()

    ea, eb = envelope(a), envelope(b)
    n = 1 << (len(ea) + len(eb)).bit_length()
    corr = np.fft.irfft(np.fft.rfft(ea, n) * np.conj(np.fft.rfft(eb, n)), n)
    lag = int(np.argmax(corr))
    if lag > n // 2:
        lag -= n
    peak = corr.max()
    confidence = float(peak / (np.abs(corr).mean() + 1e-9))
    return lag / 200.0, confidence


def cmd_sync(args):
    offset, conf = find_offset(args.camera, args.mic, seconds=args.seconds)
    ok = conf > 8
    print(f"mic starts at {offset:+.3f}s on the camera clock (confidence {conf:.1f}"
          f"{'' if ok else ', LOW: check by ear'})")
    save_json(args.out, {"camera": str(Path(args.camera).resolve()),
                         "mic": str(Path(args.mic).resolve()),
                         "offset": round(offset, 4), "confidence": round(conf, 1)})


# ---------------------------------------------------------------- transcribe

def cmd_transcribe(args):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        die("faster-whisper is not installed: pip install faster-whisper")
    model = WhisperModel(args.model, device="auto", compute_type="int8")
    # Decode with ffmpeg rather than faster-whisper's bundled reader: handles any camera format.
    audio = read_mono(args.media, 16000, 10 ** 6) / 32768.0
    segments, info = model.transcribe(
        audio, word_timestamps=True, language=args.language,
        initial_prompt=FILLER_PROMPT, vad_filter=False, condition_on_previous_text=False)
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append({"w": w.word.strip(), "start": round(w.start + args.shift, 3),
                          "end": round(w.end + args.shift, 3)})
        print(f"  {ts(seg.start)}  {seg.text.strip()}", flush=True)
    save_json(args.out, {"source": str(Path(args.media).resolve()), "language": info.language,
                         "duration": info.duration, "words": words})


# ---------------------------------------------------------------- plan: cuts

def mark_removals(words, cut_fillers=True, cut_retakes=True, retake_gap=0.25, retake_n=2):
    """Flag each word with the reason it is cut, or None to keep it."""
    reasons = [None] * len(words)
    if cut_fillers:
        for i, w in enumerate(words):
            if norm(w["w"]) in FILLERS:
                reasons[i] = "filler"

    if cut_retakes:
        # Split into phrases at pauses; if a phrase restarts the way an earlier one
        # started, the earlier attempt was a false start. Last take wins.
        phrases, cur = [], []
        for i, w in enumerate(words):
            if cur and w["start"] - words[cur[-1]]["end"] > retake_gap:
                phrases.append(cur)
                cur = []
            if reasons[i] is None:
                cur.append(i)
        if cur:
            phrases.append(cur)

        def head(p):
            return [norm(words[i]["w"]) for i in p[:retake_n]]

        for k in range(len(phrases) - 1):
            a = head(phrases[k])
            if len(a) < retake_n:
                continue
            # Look a few phrases ahead: people often stumble twice before a clean take.
            for nxt in phrases[k + 1:k + 4]:
                if head(nxt) == a:
                    for i in phrases[k]:
                        reasons[i] = "retake"
                    break
    return reasons


def build_segments(words, reasons, duration, max_pause=0.5, pad=0.12):
    """Turn kept words into source ranges, closing up long pauses and removed words."""
    kept = [(i, w) for i, w in enumerate(words) if reasons[i] is None]
    segs, cuts = [], []
    if not kept:
        return segs, cuts
    seg_start = max(kept[0][1]["start"] - pad, 0)
    prev_i, prev = kept[0]
    for i, w in kept[1:]:
        gap = w["start"] - prev["end"]
        removed = [j for j in range(prev_i + 1, i) if reasons[j]]
        if gap > max_pause or removed:
            end = prev["end"] + pad
            nxt = max(w["start"] - pad, end)
            segs.append({"start": round(seg_start, 3), "end": round(end, 3)})
            if removed:
                why = sorted({reasons[j] for j in removed})
                text = " ".join(words[j]["w"] for j in removed)
            else:
                why, text = ["pause"], ""
            cuts.append({"id": len(cuts) + 1, "removed": [round(end, 3), round(nxt, 3)],
                         "reason": "+".join(why) + f" {nxt - end:.1f}s", "text": text})
            seg_start = nxt
        prev_i, prev = i, w
    segs.append({"start": round(seg_start, 3), "end": round(min(prev["end"] + pad, duration), 3)})
    return segs, cuts


# ---------------------------------------------------------------- plan: b-roll

def load_broll_library(folder):
    folder = Path(folder)
    sidecar = folder / "broll.json"
    extra = load_json(sidecar) if sidecar.exists() else {}
    lib = []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in VIDEO_EXTS:
            continue
        tags = re.split(r"[\s_\-.]+", f.stem) + list(extra.get(f.name, []))
        tags = {stem(t) for t in tags if len(stem(t)) > 2 and not t.isdigit()}
        lib.append({"file": str(f.resolve()), "name": f.name, "tags": sorted(tags),
                    "duration": probe(f)["duration"], "used": 0, "cursor": 0.0})
    return lib


def to_record(segments):
    """Map source time -> record (timeline) time for the kept segments."""
    table, t = [], 0.0
    for s in segments:
        table.append((s["start"], s["end"], t))
        t += s["end"] - s["start"]
    return table, t


def plan_broll(words, reasons, segments, lib, lead=0.6, length=2.4, window=4.0, cover_all=True):
    table, total = to_record(segments)

    def rec(src):
        for a, b, r in table:
            if a <= src <= b:
                return r + (src - a)
        return None

    # Kept words with their timeline position, for keyword matching.
    timeline_words = []
    for i, w in enumerate(words):
        if reasons[i] is None and (r := rec((w["start"] + w["end"]) / 2)) is not None:
            timeline_words.append((r, stem(w["w"])))

    placed = []
    cut_points = [table[k][2] for k in range(1, len(table))]
    for cut in cut_points:
        start = max(cut - lead, 0)
        end = min(start + length, total)
        if placed and start < placed[-1]["record_end"]:
            # Cuts close together: stretch the clip already covering this area.
            p = placed[-1]
            room = p["clip_duration"] - p["src_in"]
            p["record_end"] = round(min(max(p["record_end"], end), p["record_start"] + room, total), 3)
            if p["record_end"] >= cut + 0.2:
                continue
            start = p["record_end"]
            end = min(start + length, total)
        if end - start < 0.5:
            continue
        nearby = {s for r, s in timeline_words if abs(r - cut) <= window}
        best, best_score = None, 0
        for clip in lib:
            score = len(nearby & set(clip["tags"])) * 10 - clip["used"] * 12
            if clip["duration"] < 0.8:
                continue
            if score > best_score or best is None:
                best, best_score = clip, score
        matched = bool(nearby & set(best["tags"])) if best else False
        if best is None or (not matched and not cover_all):
            continue
        dur = min(end - start, best["duration"])
        if best["cursor"] + dur > best["duration"]:
            best["cursor"] = 0.0
        hits = sorted(nearby & set(best["tags"]))
        placed.append({"file": best["file"], "name": best["name"],
                       "record_start": round(start, 3), "record_end": round(start + dur, 3),
                       "src_in": round(best["cursor"], 3), "clip_duration": best["duration"],
                       "reason": f"matches '{', '.join(hits)}'" if hits else "covers jump cut (no keyword match)"})
        best["cursor"] += dur
        best["used"] += 1
    for p in placed:
        p["record_end"] = round(min(p["record_end"], p["record_start"] + p["clip_duration"] - p["src_in"]), 3)
    return placed


def cmd_plan(args):
    t = load_json(args.transcript)
    words = t["words"]
    main = probe(args.main or t["source"])
    reasons = mark_removals(words, cut_fillers=not args.keep_fillers, cut_retakes=not args.keep_retakes)
    segments, cuts = build_segments(words, reasons, main["duration"], args.max_pause, args.pad)

    audio = None
    if args.sync:
        s = load_json(args.sync)
        audio = {"file": s["mic"], "offset": s["offset"]}
    lib = load_broll_library(args.broll) if args.broll else []
    broll = plan_broll(words, reasons, segments, lib, cover_all=not args.matched_only,
                       length=args.broll_length) if lib else []

    _, total = to_record(segments)
    edl = {"main": main["path"], "audio": audio, "fps": main["fps"] or 30,
           "width": main["width"] or 1920, "height": main["height"] or 1080,
           "source_duration": main["duration"], "edited_duration": round(total, 3),
           "segments": segments, "cuts": cuts, "broll": broll}
    save_json(args.out, edl)
    print(f"{ts(main['duration'])} -> {ts(total)}  ({len(cuts)} cuts, {len(broll)} b-roll clips)")


# ---------------------------------------------------------------- report

def cmd_report(args):
    edl = load_json(args.edl)
    print(f"# Rough cut: {Path(edl['main']).name}\n")
    print(f"{ts(edl['source_duration'])} of footage -> {ts(edl['edited_duration'])} edited\n")
    print("## Cuts\n")
    for c in edl["cuts"]:
        a, b = c["removed"]
        txt = f' "{c["text"]}"' if c["text"] else ""
        print(f"{c['id']:>3}. {ts(a)}-{ts(b)}  {c['reason']}{txt}")
    if edl["broll"]:
        print("\n## B-roll (timeline time)\n")
        for b in edl["broll"]:
            print(f"  {ts(b['record_start'])}-{ts(b['record_end'])}  {b['name']}  ({b['reason']})")


# ---------------------------------------------------------------- xml (FCP7 / xmeml, imports into Premiere)

def rate_xml(fps):
    ntsc = abs(fps - round(fps)) > 0.01
    base = round(fps * 1.001) if ntsc else round(fps)
    return f"<rate><timebase>{base}</timebase><ntsc>{'TRUE' if ntsc else 'FALSE'}</ntsc></rate>"


def path_url(path, media_root):
    p = Path(path)
    if media_root:
        p = Path(media_root) / p.name
    return "file://localhost" + quote(str(p).replace("\\", "/") if str(p).startswith("/") else "/" + str(p).replace("\\", "/"))


class XmlWriter:
    def __init__(self, fps, media_root):
        self.fps, self.media_root = fps, media_root
        self.files, self.n = {}, 0

    def f(self, sec):
        return int(round(sec * self.fps))

    def file_ref(self, info):
        key = info["path"]
        if key in self.files:
            return f'<file id="{self.files[key]}"/>'
        fid = f"file-{len(self.files) + 1}"
        self.files[key] = fid
        media = ""
        if info.get("width"):
            media += (f"<video><samplecharacteristics>{rate_xml(info['fps'] or self.fps)}"
                      f"<width>{info['width']}</width><height>{info['height']}</height>"
                      f"</samplecharacteristics></video>")
        if info.get("channels"):
            media += (f"<audio><samplecharacteristics><depth>16</depth><samplerate>{info['samplerate']}"
                      f"</samplerate></samplecharacteristics><channelcount>{info['channels']}</channelcount></audio>")
        return (f'<file id="{fid}"><name>{escape(Path(key).name)}</name>'
                f"<pathurl>{escape(path_url(key, self.media_root))}</pathurl>{rate_xml(info['fps'] or self.fps)}"
                f"<duration>{self.f(info['duration'])}</duration><media>{media}</media></file>")

    def clip(self, info, rec_in, rec_out, src_in, kind, track_index=1, links=()):
        self.n += 1
        cid = f"clipitem-{self.n}"
        src_out = src_in + (rec_out - rec_in)
        link_xml = "".join(
            f"<link><linkclipref>{l}</linkclipref><mediatype>{m}</mediatype><trackindex>{t}</trackindex>"
            f"<clipindex>{c}</clipindex></link>" for l, m, t, c in links)
        st = f"<sourcetrack><mediatype>{kind}</mediatype>"
        st += f"<trackindex>{track_index}</trackindex></sourcetrack>" if kind == "audio" else "</sourcetrack>"
        xml = (f'<clipitem id="{cid}"><name>{escape(Path(info["path"]).name)}</name><enabled>TRUE</enabled>'
               f"<duration>{self.f(info['duration'])}</duration>{rate_xml(self.fps)}"
               f"<start>{self.f(rec_in)}</start><end>{self.f(rec_out)}</end>"
               f"<in>{self.f(src_in)}</in><out>{self.f(src_in) + self.f(rec_out) - self.f(rec_in)}</out>"
               f"{self.file_ref(info)}{st}{link_xml}</clipitem>")
        return cid, xml


def build_xml(edl, name, media_root=None):
    fps = edl["fps"]
    w = XmlWriter(fps, media_root)
    main = probe(edl["main"])
    audio_src = probe(edl["audio"]["file"]) if edl.get("audio") else main
    offset = edl["audio"]["offset"] if edl.get("audio") else 0.0
    n_audio = max(1, min(audio_src["channels"] or 1, 2))

    v1, v2, a_tracks = [], [], [[] for _ in range(n_audio)]
    rec = 0.0
    for k, s in enumerate(edl["segments"]):
        dur = s["end"] - s["start"]
        ids = [f"clipitem-{w.n + 1 + j}" for j in range(1 + n_audio)]
        links = [(ids[0], "video", 1, k + 1)] + [(ids[1 + j], "audio", j + 1, k + 1) for j in range(n_audio)]
        _, xml = w.clip(main, rec, rec + dur, s["start"], "video", links=links)
        v1.append(xml)
        for j in range(n_audio):
            src_in = s["start"] - offset
            if src_in < 0:
                print(f"warning: segment at {ts(s['start'])} starts before the mic recording", file=sys.stderr)
                src_in = 0
            _, xml = w.clip(audio_src, rec, rec + dur, src_in, "audio", track_index=j + 1, links=links)
            a_tracks[j].append(xml)
        rec += dur

    for b in edl.get("broll", []):
        info = probe(b["file"])
        _, xml = w.clip(info, b["record_start"], b["record_end"], b["src_in"], "video")
        v2.append(xml)

    total = w.f(rec)
    r = rate_xml(fps)
    fmt = (f"<format><samplecharacteristics>{r}<width>{edl['width']}</width><height>{edl['height']}</height>"
           f"<pixelaspectratio>square</pixelaspectratio></samplecharacteristics></format>")
    track = lambda items: f"<track>{''.join(items)}<enabled>TRUE</enabled><locked>FALSE</locked></track>"
    video = fmt + track(v1) + (track(v2) if v2 else "")
    audio = (f"<numOutputChannels>2</numOutputChannels><format><samplecharacteristics><depth>16</depth>"
             f"<samplerate>48000</samplerate></samplecharacteristics></format>" + "".join(track(t) for t in a_tracks))
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n<xmeml version="4">'
            f'<sequence id="sequence-1"><name>{escape(name)}</name><duration>{total}</duration>{r}'
            f"<timecode>{r}<string>00:00:00:00</string><frame>0</frame><displayformat>NDF</displayformat></timecode>"
            f"<media><video>{video}</video><audio>{audio}</audio></media></sequence></xmeml>\n")


def cmd_xml(args):
    edl = load_json(args.edl)
    name = args.name or Path(edl["main"]).stem + " rough cut"
    Path(args.out).write_text(build_xml(edl, name, args.media_root))
    print(f"wrote {args.out}  (Premiere: File > Import, pick the .xml)")


# ---------------------------------------------------------------- preview

def cmd_preview(args):
    edl = load_json(args.edl)
    W, H = args.width, int(round(args.width * edl["height"] / edl["width"] / 2) * 2)
    inputs = ["-i", edl["main"]]
    audio_in, offset = 0, 0.0
    if edl.get("audio"):
        inputs += ["-i", edl["audio"]["file"]]
        audio_in, offset = 1, edl["audio"]["offset"]
    first_broll = len(inputs) // 2
    for b in edl["broll"]:
        inputs += ["-i", b["file"]]

    parts, labels = [], ""
    for k, s in enumerate(edl["segments"]):
        parts.append(f"[0:v]trim={s['start']}:{s['end']},setpts=PTS-STARTPTS,scale={W}:{H},fps=30[v{k}]")
        a0, a1 = max(s["start"] - offset, 0), max(s["end"] - offset, 0)
        parts.append(f"[{audio_in}:a]atrim={a0}:{a1},asetpts=PTS-STARTPTS,aresample=48000[a{k}]")
        labels += f"[v{k}][a{k}]"
    parts.append(f"{labels}concat=n={len(edl['segments'])}:v=1:a=1[base][aout]")
    last = "base"
    for j, b in enumerate(edl["broll"]):
        d = b["record_end"] - b["record_start"]
        parts.append(f"[{first_broll + j}:v]trim={b['src_in']}:{b['src_in'] + d},setpts=PTS-STARTPTS+{b['record_start']}/TB,"
                     f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps=30[b{j}]")
        parts.append(f"[{last}][b{j}]overlay=enable='between(t,{b['record_start']},{b['record_end']})':eof_action=pass[o{j}]")
        last = f"o{j}"
    graph = Path(args.out).with_suffix(".filter.txt")
    graph.write_text(";\n".join(parts))
    cmd = ["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex_script", str(graph),
           "-map", f"[{last}]", "-map", "[aout]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
           "-c:a", "aac", "-b:a", "128k", args.out]
    subprocess.run(cmd, check=True)
    graph.unlink()
    print(f"wrote {args.out}")


# ---------------------------------------------------------------- cli

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sync", help="line up a separate mic recording with camera audio")
    s.add_argument("camera"); s.add_argument("mic")
    s.add_argument("--seconds", type=float, default=600, help="how much audio to compare (default 600)")
    s.add_argument("-o", "--out", default="sync.json")
    s.set_defaults(fn=cmd_sync)

    s = sub.add_parser("transcribe", help="word-timed transcript")
    s.add_argument("media")
    s.add_argument("--model", default="small", help="tiny|base|small|medium|large-v3 (default small)")
    s.add_argument("--language", default=None)
    s.add_argument("--shift", type=float, default=0.0, help="add this many seconds to every timestamp")
    s.add_argument("-o", "--out", default="transcript.json")
    s.set_defaults(fn=cmd_transcribe)

    s = sub.add_parser("plan", help="choose cuts and b-roll")
    s.add_argument("transcript")
    s.add_argument("--main", help="camera file (default: the file that was transcribed)")
    s.add_argument("--sync", help="sync.json: use the mic recording for audio")
    s.add_argument("--broll", help="folder of b-roll clips (file names are used as tags)")
    s.add_argument("--max-pause", type=float, default=0.5, help="pauses longer than this are cut (s)")
    s.add_argument("--pad", type=float, default=0.12, help="breathing room kept around words (s)")
    s.add_argument("--broll-length", type=float, default=2.4, help="target b-roll clip length (s)")
    s.add_argument("--matched-only", action="store_true", help="only place b-roll that matches the words")
    s.add_argument("--keep-fillers", action="store_true")
    s.add_argument("--keep-retakes", action="store_true")
    s.add_argument("-o", "--out", default="edl.json")
    s.set_defaults(fn=cmd_plan)

    s = sub.add_parser("xml", help="write Premiere-importable XML")
    s.add_argument("edl")
    s.add_argument("--media-root", help="folder where the media lives on the editing machine")
    s.add_argument("--name", help="sequence name")
    s.add_argument("-o", "--out", default="roughcut.xml")
    s.set_defaults(fn=cmd_xml)

    s = sub.add_parser("preview", help="render a quick low-res preview")
    s.add_argument("edl")
    s.add_argument("--width", type=int, default=960)
    s.add_argument("-o", "--out", default="preview.mp4")
    s.set_defaults(fn=cmd_preview)

    s = sub.add_parser("report", help="list every cut and b-roll placement")
    s.add_argument("edl")
    s.set_defaults(fn=cmd_report)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
