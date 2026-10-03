"""转身片段显示节拍门禁（mac 转身顿挫修复的回归锁）。

背景：旧实现片段 media 时间按真实 dt·rate 推进、帧号 index_at(_t) 直取。
spec clip_rate=1.25 → 实播 37.5fps（帧周期 26.7ms），在 16~18ms 逻辑拍下
「帧周期/拍周期」≈1.5~1.7 非整数 → 帧边界相对拍相位漂移，显示帧长在
1 拍/2 拍间交替（mac QTimer 实测拍 ~17.7ms → 17/35ms 交替顿挫；win
15.6ms 拍 → 31/47ms 混合，同样不均——QA 抓帧门禁未覆盖节拍均匀性）。
修复：进片段时定「每帧持有 N 拍」（hold=round(帧周期/(rate·拍周期))），
media 每拍等量前进 frame_s/hold，帧边界恒落整数拍。

门禁（纯逻辑，无 Qt）：
1. mac 拍频（17.7ms）与 win 拍频（15.6ms）：中段每帧恰好 2 拍，无 1 拍帧；
2. 60Hz 理想拍：同样均匀 2 拍（旧实现在此拍频下也是 1/2 拍交替——
   顿挫并非 mac 独有，只是拍周期不同导致幅度不同）；
3. 片段总时长符合按拍预期（hold=2 → ~2 拍/帧 × 帧数），退出条件正常；
4. 交叉淡化语义保留：首端垫正面骨骼、尾端垫侧身骨骼、首尾 alpha 过 0↔1。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pet.rig.side_locomotion import LocoState, SideLocomotion, TurnClip  # noqa: E402

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f"  [{detail}]" if detail else ""), flush=True)


def make_loco() -> tuple[SideLocomotion, TurnClip]:
    pkg = ROOT / "assets" / "rig_adult_walk_v1"
    with open(pkg / "spec.json", "r", encoding="utf-8") as f:
        side_spec = json.load(f)
    out_clip = TurnClip(str(pkg / "clips" / "turn_front_to_side"))
    in_clip = TurnClip(str(pkg / "clips" / "turn_side_to_front"))
    return SideLocomotion(side_spec, out_clip, in_clip, 1.0), out_clip


def run_turn_out(dt: float, loco: SideLocomotion):
    """完整 turn_out 会话：返回（每帧持有拍数序列, TURN_OUT 总拍数,
    淡化窗内 (media 时刻, 帧号) 序列, alpha/under 轨迹）。"""
    holds, fade, alphas, unders = [], [], [], []
    wx = 100.0
    cur_idx, n = None, 0
    ticks = 0
    for _ in range(int(6.0 / dt)):
        lf = loco.update(dt, 120.0, wx)
        if lf.state is LocoState.TURN_OUT:
            ticks += 1
            f = max(loco.crossfade_s, 1e-6)
            t_media = (ticks - 1) * loco._clip_step
            if t_media < 2 * f or t_media > lf.clip.duration - 2 * f:
                fade.append((t_media, lf.clip_index))
                cur_idx, n = None, 0
            elif lf.clip_index == cur_idx:
                n += 1
            else:
                if cur_idx is not None:
                    holds.append(n)
                cur_idx, n = lf.clip_index, 1
            alphas.append((round(lf.clip_alpha, 3), lf.under))
        elif ticks:
            holds.append(n) if cur_idx is not None else None
            break
    return holds, ticks, fade, alphas


def main() -> None:
    for label, dt in (("mac 17.7ms", 0.0177), ("win 15.6ms", 0.0156), ("60Hz 16.7ms", 1 / 60)):
        print(f"== 拍频 {label}：中段持帧均匀性 ==")
        loco, _ = make_loco()
        holds, ticks, fade, alphas = run_turn_out(dt, loco)
        check(f"{label} 中段每帧恰 2 拍（无 1 拍交替帧）",
              holds and all(h == 2 for h in holds),
              f"{Counter(holds).most_common()}")

    print("== 会话时长按拍预期（hold=2 → ~2 拍/帧） ==")
    loco, clip = make_loco()
    _, ticks, _, _ = run_turn_out(1 / 60, loco)
    expect = (len(clip.frames) - 1) * 2 + 1
    check("turn_out 拍数 = 2×(帧数-1)+1（±1 拍）", abs(ticks - expect) <= 1,
          f"{ticks} 拍 vs 预期 {expect}（{len(clip.frames)} 帧）")

    print("== 交叉淡化语义保留 ==")
    loco, _ = make_loco()
    _, _, fade, alphas = run_turn_out(1 / 60, loco)
    heads = [u for _, u in alphas[:3]]
    tails = [u for _, u in alphas[-4:]]
    a_first, a_last = alphas[0][0], alphas[-1][0]
    bad_at = next((i for i, (t, idx) in enumerate(fade)
                   if idx != clip.index_at(t)), None)
    check("首端垫正面骨骼（under='front'）", all(u == "front" for u in heads), str(heads))
    check("尾端垫侧身骨骼（under='side'）", "side" in tails, str(tails))
    check("alpha 首 0 尾近 0（淡入完整；淡出由退出切换完成，旧实现同）",
          a_first == 0.0 and a_last <= 0.2,
          f"{a_first} → {a_last}")
    check("淡化窗帧号 = index_at(media 时刻)", bad_at is None and len(fade) >= 4,
          f"{len(fade)} 帧，首异位 {bad_at}")

    bad = sum(1 for _, ok, _ in results if not ok)
    print(f"\n== 门禁结果：{len(results) - bad} 通过 / {bad} 失败 ==")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
