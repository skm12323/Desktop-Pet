"""Wan 3.0 first+last-frame video via Alibaba Cloud Model Studio (DashScope) HTTP API.

The API key is read from the DASHSCOPE_API_KEY environment variable only (never stored or
logged). Endpoint: Beijing region (dashscope.aliyuncs.com, still valid next to the
workspace-specific domains). Frames are sent as base64 data URIs. Creates one async task,
polls it, downloads the mp4 (the URL expires after 24 h) and writes a JSON record next to it.
One job at a time.

Usage:
  set DASHSCOPE_API_KEY=...   (env only)
  D:\\anaconda3\\python.exe -X utf8 tools/wan3_flf2v_api.py --first F.png --last L.png \
      --prompt "..." --out assets/rig_adult_walk_v1/clips/raw/wan3/try1.mp4 [--resolution 1080P]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://dashscope.aliyuncs.com/api/v1"


def _req(method: str, url: str, key: str, body: dict | None = None, extra: dict | None = None) -> dict:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **(extra or {})}
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        return json.loads(urllib.request.urlopen(req, timeout=120).read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:1500]}")


def _data_uri(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", required=True)
    ap.add_argument("--last", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="wan3.0-video")
    ap.add_argument("--resolution", default="1080P")
    ap.add_argument("--ratio", default="9:16")
    ap.add_argument("--duration", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=3000)
    a = ap.parse_args()
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not key:
        raise SystemExit("set DASHSCOPE_API_KEY in the environment")
    first, last, out = Path(a.first), Path(a.last), Path(a.out)
    params = {"resolution": a.resolution, "ratio": a.ratio, "duration": a.duration, "audio": False,
              "seed": a.seed, "prompt_extend": False, "watermark": False}
    body = {"model": a.model,
            "input": {"prompt": a.prompt, "media": [{"type": "first_frame", "url": _data_uri(first)},
                                                    {"type": "last_frame", "url": _data_uri(last)}]},
            "parameters": params}
    t0 = time.time()
    r = _req("POST", f"{BASE}/services/aigc/video-generation/video-synthesis", key, body, {"X-DashScope-Async": "enable"})
    tid = r["output"]["task_id"]
    print("task", tid, r["output"]["task_status"], flush=True)
    info = {}
    while time.time() - t0 < a.timeout:
        time.sleep(15)
        info = _req("GET", f"{BASE}/tasks/{tid}", key)
        st = info["output"]["task_status"]
        print(f"{time.time() - t0:5.0f}s {st}", flush=True)
        if st in ("SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"):
            break
    if info.get("output", {}).get("task_status") != "SUCCEEDED":
        raise SystemExit("failed: " + json.dumps(info, ensure_ascii=False)[:1500])
    out.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(info["output"]["video_url"], out)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    rec = {"tool": "tools/wan3_flf2v_api.py", "endpoint": BASE, "model": a.model, "task_id": tid,
           "seconds": round(time.time() - t0, 1), "parameters": params, "prompt": a.prompt,
           "first": {"path": first.as_posix(), "sha256": sha(first)},
           "last": {"path": last.as_posix(), "sha256": sha(last)},
           "usage": info.get("usage"), "output": out.as_posix(), "output_sha256": sha(out)}
    out.with_suffix(".json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": rec["output"], "usage": rec["usage"], "seconds": rec["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
