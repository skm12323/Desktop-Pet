"""G3: generated green-screen turn clip -> keyed, registered, retimed RGBA frame package.

Pipeline (docs/ADULT转身视频过渡与侧身骨骼行走方案-2026-09-27.md §3 G3):
  decode (ffmpeg) -> colour-difference key on #00B140 + despill (native clip resolution)
  -> register into the output space (extended rig canvas scaled to --height px) through the
     endpoint mapping -> clean alpha (drop specks, clear below the ground line)
  -> ONE global colour transform per clip, fitted jointly on both endpoints
  -> retime (source frame range, output duration, nearest-frame sampling)
  -> endpoint morph: the first/last --morph-frames output frames are flow-warped and
     dissolved onto the true rig renders, so the clip starts exactly on the front rest pose
     and ends exactly on the side rest pose (no seam, no pop at the rig hand-over)
  -> per-frame cropped straight-RGBA PNGs + clip.json.

Output space: pixel (u, v) <-> rig canvas (x, y) = (u / scale + origin_x, v / scale + origin_y);
each PNG is a crop whose top-left is stored per frame in clip.json.

Usage (repo root):
  D:\\anaconda3\\python.exe -X utf8 tools/process_turn_clip.py \\
      assets/rig_adult_walk_v1/clips/raw/wan/dryrun4_l160r40_s1.mp4 \\
      --endpoints assets/rig_adult_walk_v1/clips/endpoints/front_to_side_l160r40 \\
      --first-ref assets/rig_adult_walk_v1/references/front_rest.png \\
      --last-ref  assets/rig_adult_walk_v1/references/side_key.png \\
      --out assets/rig_adult_walk_v1/clips/turn_front_to_side --duration 1.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spikes"))
from qa_turn_clip import decode, key_frame  # noqa: E402
from polish_turn_frames import polish_rgba  # noqa: E402

CANVAS = (960, 1696)
GROUND_Y = 1608.0


def resize_box(rgba: np.ndarray, size: tuple[int, int], box: tuple[float, float, float, float]) -> np.ndarray:
    """Straight RGBA float -> premultiplied RGBA float, Lanczos-resampled from a (float) source box."""
    pm = rgba.astype(np.float32).copy()
    pm[..., :3] *= pm[..., 3:4] / 255.0
    ch = [np.asarray(Image.fromarray(pm[..., i]).resize(size, Image.Resampling.LANCZOS, box=box))
          for i in range(4)]
    out = np.stack(ch, -1)
    out[..., 3] = np.clip(out[..., 3], 0, 255)
    out[..., :3] = np.clip(out[..., :3], 0, out[..., 3:4])
    return out


def unpremul(pm: np.ndarray) -> np.ndarray:
    a = pm[..., 3:4]
    rgb = np.where(a > 0.5, pm[..., :3] * 255.0 / np.maximum(a, 1e-3), 0.0)
    return np.dstack([np.clip(rgb, 0, 255), a])


class OutputSpace:
    """Extended rig canvas (x from the video frame's left edge) scaled to `height` px."""

    def __init__(self, mapping: dict, height: int):
        self.s = height / CANVAS[1]
        self.ox = float(mapping["frame_origin_canvas"][0])
        self.oy = 0.0
        fw = mapping["frame_size_canvas"][0]
        self.size = (int(round(fw * self.s)), height)
        self.m = mapping

    def video_box(self, clip_w: int, clip_h: int) -> tuple[float, float, float, float]:
        """Box in clip pixels covering the output space. The model input was the endpoint
        resized to the clip size (e.g. 1080x1920 -> 480x848, not exactly 9:16), so x and y
        scale separately."""
        sx = self.m["scale_video_per_canvas"] * clip_w / self.m["video_size"][0]
        sy = self.m["scale_video_per_canvas"] * clip_h / self.m["video_size"][1]
        x0v, y0v = self.m["frame_origin_canvas"]
        cx0, cy0 = self.ox, self.oy
        cx1, cy1 = self.ox + self.size[0] / self.s, self.oy + self.size[1] / self.s
        return ((cx0 - x0v) * sx, (cy0 - y0v) * sy, (cx1 - x0v) * sx, (cy1 - y0v) * sy)

    def from_canvas(self, rgba_canvas: np.ndarray) -> np.ndarray:
        box = (self.ox, self.oy, self.ox + self.size[0] / self.s, self.oy + self.size[1] / self.s)
        pad = int(np.ceil(-self.ox)) + 2
        ext = np.zeros((CANVAS[1], CANVAS[0] + 2 * pad, 4), np.float32)
        ext[:, pad:pad + CANVAS[0]] = rgba_canvas
        return resize_box(ext, self.size, (box[0] + pad, box[1], box[2] + pad, box[3]))

    def ground_row(self) -> float:
        return (GROUND_Y - self.oy) * self.s


def clean_alpha(pm: np.ndarray, ground_row: float) -> np.ndarray:
    """Keep the character's connected component(s); nothing below the ground line."""
    a = pm[..., 3] / 255.0
    lab, n = ndimage.label(a > 0.08)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(a), lab, range(1, n + 1))
        keep = np.isin(lab, 1 + np.where(sizes >= max(sizes.max() * 0.02, 30))[0])
        keep = ndimage.binary_dilation(keep, iterations=2)
        pm = pm * keep[..., None]
    g = int(np.ceil(ground_row + 3))
    pm[g:] = 0.0
    return pm


def fit_colour(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel gain/offset on straight colour where both are opaque, jointly over pairs."""
    xs, ys = [[], [], []], [[], [], []]
    for src, ref in pairs:
        s, r = unpremul(src), unpremul(ref)
        m = (s[..., 3] > 240) & (r[..., 3] > 240)
        for c in range(3):
            xs[c].append(s[..., c][m])
            ys[c].append(r[..., c][m])
    gains, offs = np.ones(3, np.float32), np.zeros(3, np.float32)
    for c in range(3):
        x, y = np.concatenate(xs[c]), np.concatenate(ys[c])
        A = np.stack([x, np.ones_like(x)], 1)
        (gains[c], offs[c]), *_ = np.linalg.lstsq(A, y, rcond=None)
    return gains, offs


def apply_colour(pm: np.ndarray, gains: np.ndarray, offs: np.ndarray) -> np.ndarray:
    s = unpremul(pm)
    s[..., :3] = np.clip(s[..., :3] * gains + offs, 0, 255)
    out = s.copy()
    out[..., :3] *= s[..., 3:4] / 255.0
    return out


def flow_to(ref_pm: np.ndarray, mov_pm: np.ndarray) -> np.ndarray:
    """Dense flow f (2,H,W) with mov(x + f(x)) ~= ref(x); luma on mid-grey so edges count."""
    from skimage.registration import optical_flow_tvl1

    def gray(pm):
        a = pm[..., 3:4] / 255.0
        rgb = pm[..., :3] + (1.0 - a) * 128.0
        return (rgb @ np.array([0.299, 0.587, 0.114], np.float32)) / 255.0
    return optical_flow_tvl1(gray(ref_pm), gray(mov_pm), attachment=15, tightness=0.3,
                             num_warp=8, num_iter=20)


def warp(pm: np.ndarray, flow: np.ndarray, t: float) -> np.ndarray:
    """Sample pm at x + t * flow(x) (bilinear, premultiplied channels)."""
    h, w = pm.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    coords = [yy + t * flow[0], xx + t * flow[1]]
    return np.stack([ndimage.map_coordinates(pm[..., c], coords, order=1, mode="constant")
                     for c in range(4)], -1)


def morph_onto(frame_pm: np.ndarray, truth_pm: np.ndarray, w: float) -> np.ndarray:
    """w=0 -> frame unchanged, w=1 -> truth exactly; in between flow-aligned dissolve."""
    if w <= 0:
        return frame_pm
    if w >= 1:
        return truth_pm.copy()
    f = flow_to(truth_pm, frame_pm)            # frame(x + f) ~= truth(x)
    frame_w = warp(frame_pm, f, w)              # frame moved w of the way onto truth geometry
    truth_w = warp(truth_pm, -f, 1.0 - w)       # truth moved back towards the frame geometry
    return (1.0 - w) * frame_w + w * truth_w


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def premul_diff(a: np.ndarray, b: np.ndarray) -> float:
    sil = (a[..., 3] > 8) | (b[..., 3] > 8)
    return float(np.abs(a[..., :3] - b[..., :3]).mean(-1)[sil].mean())


def crop_bbox(pm: np.ndarray, pad: int = 2) -> tuple[int, int, int, int]:
    ys, xs = np.where(pm[..., 3] > 0.5)
    h, w = pm.shape[:2]
    return (int(max(0, xs.min() - pad)), int(max(0, ys.min() - pad)),
            int(min(w, xs.max() + 1 + pad)), int(min(h, ys.max() + 1 + pad)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clip")
    ap.add_argument("--endpoints", required=True, help="dir with mapping.json (prepare_video_endpoints.py)")
    ap.add_argument("--first-ref", required=True, help="canvas RGBA the clip must start on (rig rest render)")
    ap.add_argument("--last-ref", required=True, help="canvas RGBA the clip must end on (rig rest render)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--src-range", default="", help="first:last source frame (inclusive), default whole clip")
    ap.add_argument("--duration", type=float, default=1.0, help="seconds for the source range (morph holds add 2*(k+1)-2 frames)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--height", type=int, default=512, help="output-space height = 2x of the 256 window")
    ap.add_argument("--hold-frames", type=int, default=3,
                    help="pure rest frames added at each end (runtime crossfade runs over them)")
    ap.add_argument("--morph-frames", type=int, default=3, help="static morph frames added at each end (truth <-> source endpoint)")
    ap.add_argument("--motion-retime", type=float, default=0.0,
                    help="0 = uniform; 0..1 = share of time spread evenly, rest follows frame-to-frame motion")
    ap.add_argument("--reverse", action="store_true", help="play the source backwards (diagnostic only)")
    ap.add_argument("--qa-dir", default="", help="evidence dir (default spikes/_qa/adult_walk_v1/g3_turn_clip/<out name>)")
    a = ap.parse_args()
    qa = Path(a.qa_dir or f"spikes/_qa/adult_walk_v1/g3_turn_clip/{Path(a.out).name}")
    qa.mkdir(parents=True, exist_ok=True)

    out = Path(a.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    for old in (out / "frames").glob("*.png"):
        old.unlink()
    mapping = json.loads((Path(a.endpoints) / "mapping.json").read_text(encoding="utf-8"))["first"]
    space = OutputSpace(mapping, a.height)

    frames, src_fps = decode(a.clip)
    n_src = len(frames)
    lo, hi = (0, n_src - 1)
    if a.src_range:
        lo, hi = (int(v) for v in a.src_range.split(":"))
    n_out = max(2, int(round(a.duration * a.fps)) + 1)
    src_idx = np.rint(np.linspace(lo, hi, n_out)).astype(int)
    if a.motion_retime > 0:
        # time follows motion: cumulative (baseline + silhouette/colour change) is sampled
        # uniformly, so near-static holds shrink and the turn itself keeps its pacing
        small = [np.asarray(Image.fromarray(frames[i]).resize((96, 170), Image.Resampling.BILINEAR),
                            np.float32) for i in range(lo, hi + 1)]
        step = np.array([0.0] + [np.abs(small[i] - small[i - 1]).mean() for i in range(1, len(small))])
        w = step / max(step.sum(), 1e-6) * (1.0 - a.motion_retime) + a.motion_retime / len(step)
        cum = np.cumsum(w)
        cum = (cum - cum[0]) / (cum[-1] - cum[0])
        src_idx = lo + np.rint(np.interp(np.linspace(0, 1, n_out), cum, np.arange(len(cum)))).astype(int)
    if a.reverse:
        src_idx = src_idx[::-1]

    clip_h, clip_w = frames[0].shape[:2]
    box = space.video_box(clip_w, clip_h)
    keyed_cache: dict[int, np.ndarray] = {}

    def keyed(i: int) -> np.ndarray:
        if i not in keyed_cache:
            rgba = np.pad(key_frame(frames[i].astype(np.float32)), ((8, 8), (8, 8), (0, 0)))
            pm = resize_box(rgba, space.size, tuple(v + 8 for v in box))
            keyed_cache[i] = clean_alpha(pm, space.ground_row())
        return keyed_cache[i]

    truth_first = space.from_canvas(np.asarray(Image.open(a.first_ref).convert("RGBA"), np.float32))
    truth_last = space.from_canvas(np.asarray(Image.open(a.last_ref).convert("RGBA"), np.float32))

    gains, offs = fit_colour([(keyed(int(src_idx[0])), truth_first), (keyed(int(src_idx[-1])), truth_last)])
    raw_first = apply_colour(keyed(int(src_idx[0])), gains, offs)
    raw_last = apply_colour(keyed(int(src_idx[-1])), gains, offs)
    endpoint_before = {"first_premul_mean_abs_255": premul_diff(raw_first, truth_first),
                       "last_premul_mean_abs_255": premul_diff(raw_last, truth_last)}

    # Endpoint morphs run over static holds (truth <-> source endpoint frame), never over
    # moving frames: dissolving onto a moving clip ghosts the swinging tail.
    k = max(0, a.morph_frames)
    body = [apply_colour(keyed(int(i)), gains, offs) for i in src_idx]
    ease = lambda u: float(smoothstep(np.array(u)))   # small steps next to the rest frames
    head = [morph_onto(body[0], truth_first, ease(1.0 - j / (k + 1))) for j in range(k + 1)] if k else []
    tail = [morph_onto(body[-1], truth_last, ease((j + 1) / (k + 1))) for j in range(k + 1)] if k else []
    # pure rest frames at both ends: the runtime crossfades (rig <-> clip) run over these, so the
    # fade never overlaps the morph (dedupe below stores them once)
    head = [truth_first.copy()] * a.hold_frames + head
    tail = tail + [truth_last.copy()] * a.hold_frames
    out_frames = [clean_alpha(pm, space.ground_row()) for pm in head + body[1:-1] + tail]
    src_idx = np.concatenate([np.full(len(head), src_idx[0]), src_idx[1:-1], np.full(len(tail), src_idx[-1])])
    n_out = len(out_frames)
    if not k:
        out_frames = [clean_alpha(pm, space.ground_row()) for pm in body]
        src_idx, n_out = src_idx, len(out_frames)

    # package: straight RGBA crops (+ per-frame offsets); decoded memory = sum of crop areas
    recs, bytes_decoded = [], 0
    seen: dict = {}                               # identical frames (holds) -> one file
    for j, pm in enumerate(out_frames):
        x0, y0, x1, y1 = crop_bbox(pm)
        rgba = np.clip(unpremul(pm[y0:y1, x0:x1]) + 0.5, 0, 255).astype(np.uint8)
        keep = a.hold_frames + a.morph_frames + 1
        if keep <= j < len(out_frames) - keep:
            rgba, _ = polish_rgba(rgba, a.height)
        img = Image.fromarray(rgba, "RGBA")
        digest = hashlib.sha256(img.tobytes() + bytes([x0 % 256, y0 % 256])).hexdigest()
        if digest in seen:
            name = seen[digest]
        else:
            name = f"{j:03d}.png"
            img.save(out / "frames" / name, optimize=True)
            seen[digest] = name
            bytes_decoded += (x1 - x0) * (y1 - y0) * 4
        recs.append({"file": f"frames/{name}", "offset_px": [x0, y0], "size_px": [x1 - x0, y1 - y0],
                     "source_frame": int(src_idx[j])})
    disk = sum(p.stat().st_size for p in (out / "frames").glob("*.png"))

    endpoint_after = {"first_premul_mean_abs_255": premul_diff(out_frames[0], truth_first),
                      "last_premul_mean_abs_255": premul_diff(out_frames[-1], truth_last)}
    flick = []
    for seg in (out_frames[:3], out_frames[-3:]):
        for x, y in zip(seg, seg[1:]):
            ax, ay = x[..., 3] / 255.0, y[..., 3] / 255.0
            s_ = (ax > 0.5) | (ay > 0.5)
            flick.append(float((np.abs(ax - ay)[s_] > 0.2).mean()))
    green = np.concatenate([(unpremul(f)[..., 1] - np.maximum(unpremul(f)[..., 0], unpremul(f)[..., 2]))[f[..., 3] > 51]
                            for f in out_frames])
    clip_json = {
        "tool": "tools/process_turn_clip.py",
        "source_clip": Path(a.clip).as_posix(),
        "source_sha256": hashlib.sha256(Path(a.clip).read_bytes()).hexdigest(),
        "source_fps": src_fps, "source_frames": n_src, "source_range": [lo, hi], "reversed": a.reverse,
        "motion_retime": a.motion_retime,
        "first_ref": Path(a.first_ref).as_posix(), "last_ref": Path(a.last_ref).as_posix(),
        "fps": a.fps, "frame_count": n_out, "duration_s": (n_out - 1) / a.fps,
        "space": {"scale_per_canvas_px": space.s, "origin_canvas": [space.ox, space.oy],
                  "size_px": list(space.size),
                  "note": "canvas_xy = (frame_offset + png_uv) / scale + origin_canvas"},
        "canvas_size": list(CANVAS), "ground_y_canvas": GROUND_Y,
        "colour_transform": {"gains": gains.tolist(), "offsets": offs.tolist()},
        "morph_frames": k, "hold_frames": a.hold_frames,
        "postprocess": {"tool": "tools/polish_turn_frames.py", "version": 1,
                        "alpha_unchanged": True},
        "root_motion_canvas_px": [[0.0, 0.0]] * n_out,
        "frames": recs,
        "qa": {"endpoint_before_morph": endpoint_before, "endpoint_after_morph": endpoint_after,
               "endpoint_alpha_flicker_max": max(flick), "endpoint_alpha_flicker_pass": bool(max(flick) <= 0.003),
               "residual_green_p99_255": float(np.percentile(green, 99)),
               "disk_bytes": disk, "decoded_bytes": bytes_decoded,
               "size_pass": bool(disk <= 10e6 and bytes_decoded <= 12e6)},
    }
    (out / "clip.json").write_bytes(json.dumps(clip_json, indent=2).encode("utf-8"))

    # evidence: light/dark contact sheet at 256 + 1x / 0.25x GIFs
    tw, th = int(round(space.size[0] / 2)), a.height // 2
    idx = np.linspace(0, n_out - 1, min(12, n_out)).round().astype(int)
    sheet = Image.new("RGB", (tw * len(idx), th * 2 + 16), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    gif = []
    for j, pm in enumerate(out_frames):
        small = Image.fromarray(np.clip(unpremul(pm) + 0.5, 0, 255).astype(np.uint8), "RGBA").resize(
            (tw, th), Image.Resampling.LANCZOS)
        bg = Image.new("RGBA", (tw, th), (245, 246, 250, 255))
        bg.alpha_composite(small)
        gif.append(bg.convert("RGB"))
        if j in idx:
            k_ = list(idx).index(j)
            for row, bgc in enumerate(((245, 246, 250), (34, 40, 49))):
                b = Image.new("RGBA", (tw, th), (*bgc, 255))
                b.alpha_composite(small)
                sheet.paste(b.convert("RGB"), (k_ * tw, 16 + row * th))
            dr.text((k_ * tw + 2, 2), f"{j}<-{int(src_idx[j])}", fill=(0, 0, 0))
    sheet.save(qa / "contact_sheet_256.png")
    dur = int(round(1000 / a.fps))
    gif[0].save(qa / "clip_256_1x.gif", save_all=True, append_images=gif[1:], duration=dur, loop=0)
    gif[0].save(qa / "clip_256_quarter.gif", save_all=True, append_images=gif[1:], duration=dur * 4, loop=0)
    (qa / "metrics.json").write_bytes(json.dumps(clip_json["qa"], indent=2).encode("utf-8"))
    print(json.dumps(clip_json["qa"], indent=2))


if __name__ == "__main__":
    main()
