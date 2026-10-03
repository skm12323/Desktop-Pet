# M1 spike 报告（S2.2 第一轮，2026-10-04）

> 环境：PySide6 **6.10.3**（venv 实装高于 requirements 的 6.5–6.7 锁，生产化前需统一），
> Apple M5 / 24GB / Metal 默认后端，macOS。

## 门槛判定表

| 项 | 结果 | 判定 |
|---|---|---|
| 透明置顶无边框窗 | 壁纸从角色周围透出（合成器截图证实），QQuickView 直窗与 QQuickWidget 两条壳都通 | **✓** |
| RSS 增量 | 60k 破碎网格 +45MB；**全分辨率 72 万面 +153MB**（超 +80MB 门槛）；结论：引擎本体开销小，大头是资产——**等 Blender 正规减面（连通拓扑，~3 万面）后复测** | ◐（资产侧未收口） |
| CPU（静止 5–6s） | 0.1–0.2s（dirty 驱动停帧在 3D 成立） | **✓** |
| CPU（连续旋转 6s） | 0.26s（单核 ~4%，60fps 连续渲染的量级可接受） | **✓** |
| 画风（toon） | 未实现（S2.3），本轮用 NoLighting 剪影验证管线 | 待 S2.3 |
| 骨骼桥 | 未测（S2.4） | 待测 |

## 本轮抓到的坑（全部入册）

1. **fast-simplification 减面炸拓扑**：60k 面输出 `body_count=27381`（碎纸屑），
   渲染成"雪花洞"（Qt 默认背面剔除+碎片间隙；Blender 双面渲染故前未见）。
   **教训：AI 网格减面必须走 Blender decimate（保连通）；判定渲染问题先查
   `body_count`，别在材质/捕获路径上绕弯。**
2. `QWidget.grab()` 对半透明窗产出 alpha 噪声——截屏必须走 `QScreen.grabWindow`
   （系统合成结果=肉眼所见）。
3. QML 注释只认 `//` 不认 `#`；QWindow（QQuickView）无 `setAttribute/move`，
   用 `setColor(透明)` + `setFlags/setPosition`。
4. QQuickWidget 与 QQuickView 两条壳在 macOS 透明置顶都可用；正式桌宠窗沿用
   现有 QQuickWidget 通道时无额外风险，View3D 默认 Offscreen + Transparent 即可。

## 复跑

```bash
.venv/bin/python three_d/spikes/m1/spike_qml3d.py baseline --duration 5
.venv/bin/python three_d/spikes/m1/spike_qml3d.py view3d --shell view --duration 6 --screenshot
.venv/bin/python three_d/spikes/m1/spike_qml3d.py view3d --shell view --spin --duration 6
```

## S2.2 收口条件（剩余）

- [ ] Blender 正规减面 ~3 万面 → RSS 复测（门槛 ≤ +80MB）
- [ ] toon 材质（S2.3 顺带做）→ 画风人工判定（一票否决项）
- [ ] 骨骼桥（S2.4 顺带做）→ 逐帧驱动实测
