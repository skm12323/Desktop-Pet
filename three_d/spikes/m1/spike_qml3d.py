"""M1 spike —— 透明置顶窗 + View3D 全链路验证 + 资源实测（S2.2，three_d/wiki/方案-工作分解.md）。

双模式对照（RSS 门槛 ≤ +80MB 的测量基线）：
  python three_d/spikes/m1/spike_qml3d.py baseline   # 同窗体 flag 的空 QQuickWidget
  python three_d/spikes/m1/spike_qml3d.py view3d     # View3D + Balsam 草模
  python three_d/spikes/m1/spike_qml3d.py view3d --spin   # 附加持续旋转（最坏功耗态）

测量项：RSS 时间序列（psutil，缺则 ru_maxrss）、进程 CPU 时间、RHI 后端名
（QSG_RENDERER_API / 环境回显）、窗口是否成功透明置顶（人工目测窗口期）。
自动退出后打印 JSON 结果行（便于脚本比对）。

耗时：默认 --duration 8（秒）。窗口在桌面上短暂出现属预期。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

MESH = os.path.join(
    REPO, "three_d", "spikes", "m1", "assets", "full_mesh.qml",
    "meshes", "geometry_0_mesh.mesh",
)
SCENE = os.path.join(os.path.dirname(__file__), "spike_scene.qml")


def rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # mac: KB→MB


def cpu_seconds() -> float:
    return time.process_time()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["baseline", "view3d"])
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--spin", action="store_true")
    ap.add_argument("--screenshot", action="store_true",
                    help="运行中段抓窗口 PNG（人工判定画风/透明合成用）")
    ap.add_argument("--shell", choices=["widget", "view"], default="widget",
                    help="widget=QQuickWidget（离屏合成）/ view=QQuickView（直窗）")
    args = ap.parse_args()

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen" if os.environ.get("SPIKE_OFFSCREEN") else "cocoa")

    from PySide6.QtCore import QTimer, QUrl
    from PySide6.QtGui import QColor, Qt
    from PySide6.QtWidgets import QApplication

    app = QApplication([])

    if args.shell == "view":
        from PySide6.QtQuick import QQuickView
        win = QQuickView()
        win.setColor(QColor(0, 0, 0, 0))     # QWindow 透明：clearColor 即可
        win.setResizeMode(QQuickView.ResizeMode.SizeRootObjectToView)
        win.setFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
    else:
        from PySide6.QtQuickWidgets import QQuickWidget
        win = QQuickWidget()
        win.setClearColor(QColor(0, 0, 0, 0))
        win.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
        win.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        win.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
    win.resize(240, 420)

    scene = SCENE if args.mode == "view3d" else \
        os.path.join(os.path.dirname(__file__), "spike_blank.qml")
    if args.mode == "view3d":
        assert os.path.isfile(MESH), f"缺 Balsam 产物：{MESH}（先跑 pyside6-balsam）"
    win.setSource(QUrl.fromLocalFile(scene))
    if args.mode == "view3d":
        root = win.rootObject()
        root.setProperty("meshUrl", QUrl.fromLocalFile(MESH).toString())
        root.setProperty("spin", args.spin)

    # 移到屏幕右下角（贴近桌宠日常位置）
    screen = app.primaryScreen().availableGeometry()
    if args.shell == "view":
        win.setPosition(screen.right() - win.width() - 40,
                        screen.bottom() - win.height() - 40)
    else:
        win.move(screen.right() - win.width() - 40,
                 screen.bottom() - win.height() - 40)
    win.show()

    samples: list[dict] = []
    cpu0, t0 = cpu_seconds(), time.monotonic()

    def sample() -> None:
        samples.append({"t": round(time.monotonic() - t0, 2), "rss_mb": round(rss_mb(), 1)})

    sample()
    timer = QTimer()
    timer.setInterval(500)
    timer.timeout.connect(sample)
    timer.start()

    def finish() -> None:
        sample()
        rss0, rss1 = samples[0]["rss_mb"], max(s["rss_mb"] for s in samples)
        result = {
            "mode": args.mode,
            "spin": args.spin,
            "duration_s": args.duration,
            "rss_first_mb": rss0,
            "rss_peak_mb": rss1,
            "rss_delta_mb": round(rss1 - rss0, 1),
            "cpu_seconds": round(cpu_seconds() - cpu0, 2),
            "backend_hint": os.environ.get("QSG_RHI_BACKEND", "default"),
            "samples_tail": samples[-3:],
        }
        print("SPIKE_RESULT " + json.dumps(result, ensure_ascii=False))
        app.quit()

    QTimer.singleShot(int(args.duration * 1000), finish)
    if args.screenshot:
        def grab() -> None:
            # 进程内 QScreen.grabWindow（抓自身窗口不需要屏幕录制权限；
            # 外部 screencapture 需要 TCC 授权，本环境没有——曾致静默失败看旧图）。
            scr = app.primaryScreen()
            img = scr.grabWindow(int(win.winId()))
            out = os.path.join(os.path.dirname(__file__), f"shot_{args.mode}.png")
            img.save(out)
            print(f"SCREENSHOT {out} (mtime={os.path.getmtime(out):.0f})")
        QTimer.singleShot(int(args.duration * 1000 * 0.6), grab)
    app.exec()


if __name__ == "__main__":
    main()
