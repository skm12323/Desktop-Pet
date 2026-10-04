"""test_engine_bridge —— 中间层（接口接入）回归锁。

验证三层结构「原有引擎 ← 中间层 ← 新引擎有效部分」：
* 原有引擎缺省（NullEnricher）恒等、零风险；
* 新引擎有效部分（ChannelEnricher：风/光影）产出正确增量；
* 中间层（EngineBridge）防御性：新引擎抛错 → 永久降级恒等，不阻断不刷屏。

运行：.venv/bin/python spikes/test_engine_bridge.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pet.engine_bridge import (  # noqa: E402
    ChannelEnricher, EngineBridge, Enrichment, NullEnricher,
)

PASS = 0
FAIL = 0



def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}" + (f"  {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  ❌ {name}" + (f"  {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
def test_null_identity() -> None:
    print("[T1] NullEnricher 恒等 + 零风险")
    n = NullEnricher()
    n.set_motion(tilt_deg=9.0, walking=True, airborne=False)
    e = n.tick(33.0)
    check("恒等 body_angle=0", e.body_angle == 0.0)
    check("恒等 body_y=0", e.body_y == 0.0)
    check("恒等 scale=1", e.scale_x == 1.0 and e.scale_y == 1.0)
    check("恒等 blink=False", e.blink_on is False)
    check("恒等 shadow=0", e.shadow_alpha == 0.0)


def test_channel_enricher() -> None:
    print("[T4] ChannelEnricher 出风调制 + 阴影（静态兜底，确定性）")
    # enabled=False → StaticWindSource(1.0) / StaticSunSource(0.35)，无网络无时变
    cfg = {"wind": {"enabled": False},
           "sun": {"enabled": False, "shadow_alpha": 0.35}}
    c = ChannelEnricher(cfg)
    gain, bias = c.wind()
    sh = c.shadow()
    check("风增益数值", gain > 0.0, f"gain={gain:.2f}")
    check("顺风偏置有值", isinstance(bias, float))
    check("阴影有 alpha", sh.shadow_alpha > 0.0, f"alpha={sh.shadow_alpha:.2f}")


def test_bridge_original_fallback() -> None:
    print("[T5] EngineBridge(Null) → 恒等 + active=original")
    b = EngineBridge(None)
    b.set_motion(tilt_deg=5.0, walking=True)
    e = b.tick(33.0)
    check("active=original", b.active == "original")
    check("恒等输出", e.body_angle == 0.0 and e.scale_y == 1.0)
    check("无通道 wind() 回退 (1.0, 0.0)", b.wind() == (1.0, 0.0))


class _BoomEnricher(NullEnricher):
    def set_motion(self, **kw):
        raise RuntimeError("boom")

    def tick(self, dt_ms):
        raise RuntimeError("boom")


def test_bridge_defensive_degrade() -> None:
    print("[T7] EngineBridge 防御性：抛错 → 永久降级恒等，不阻断")
    b = EngineBridge(_BoomEnricher())
    b.set_motion(tilt_deg=1.0)                # 内部抛错，应被吞掉
    e = b.tick(33.0)                          # 内部抛错，应被吞掉
    check("降级标记", b.degraded is True)
    check("active 回 original", b.active == "original")
    check("tick 仍返回恒等", e.body_angle == 0.0)
    check("不阻断（无异常外抛）", True)


def test_bridge_full_shadow() -> None:
    print("[T8] EngineBridge(Null + channels) → 恒等运动 + 阴影叠加（rig 后端装配形态）")
    cfg = {"sun": {"enabled": False, "shadow_alpha": 0.35}}   # 静态阴影，确定性
    b = EngineBridge(NullEnricher(), ChannelEnricher(cfg))
    b.refresh_channels()
    wg, wb = b.wind()
    check("wind() 取值（静态微风）", wg > 0.0 and isinstance(wb, float),
          f"gain={wg:.2f}")
    b.set_motion(tilt_deg=3.0)
    e = b.tick(33.0)
    check("运动恒等（rig 自持 MotionEngine）", e.body_angle == 0.0 and e.scale_y == 1.0)
    check("阴影被叠加", e.shadow_alpha > 0.0, f"alpha={e.shadow_alpha:.2f}")


def main() -> int:
    test_null_identity()
    test_channel_enricher()
    test_bridge_original_fallback()
    test_bridge_defensive_degrade()
    test_bridge_full_shadow()
    print(f"\nengine_bridge 中间层: {PASS} 通过, {FAIL} 失败")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
