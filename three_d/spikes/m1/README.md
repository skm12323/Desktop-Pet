# M1 spike 报告（S2.2 第一轮，2026-10-04）

> 环境：PySide6 **6.10.3**（venv 实装高于 requirements 的 6.5–6.7 锁，生产化前需统一），
> Apple M5 / 24GB / Metal 默认后端，macOS。

## 门槛判定表

| 项 | 结果 | 判定 |
|---|---|---|
| 透明置顶无边框窗 | 壁纸从角色周围透出（合成器截图证实），QQuickView 直窗与 QQuickWidget 两条壳都通 | **✓** |
| RSS 增量 | **三点实测**：60k 破碎网格 +45MB / **TRELLIS 20 万面 +79.3MB** / 全分辨率 72 万面 +153MB——线性于面数，**正规 3 万面连通网格预计 +40~50MB，门槛 ≤+80MB 无虞**；引擎本体开销小，大头是资产 | ◐→✓（缩放规律已证实，正式减面后复测收口） |
| CPU（静止 5–6s） | 0.1–0.2s（dirty 驱动停帧在 3D 成立） | **✓** |
| CPU（连续旋转 6s） | 0.26s（单核 ~4%，60fps 连续渲染的量级可接受） | **✓** |
| 画风（toon 首刀） | **✓ 机制成立**（2026-10-04）：CustomMaterial 零参数钩子（DIFFUSE/LIGHT_COLOR/TO_LIGHT_DIR/VAR_WORLD_NORMAL）+ 阶梯两段 + rim 边缘光 + 基色自乘进光照累积；蓝紫基色/分带/轮廓特征截图验证。**最终画风判定等 S1.1 正式网格**（草模法线噪声仍在） | ✓（S2.3 持续调参） |
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
5. **CustomMaterial 三个坑**：(a) 钩子签名是**零参数**（无 LIGHT light 参数），
   光数据走扁平变量 LIGHT_COLOR/TO_LIGHT_DIR，法线是 VAR_WORLD_NORMAL；
   (b) 无独立 ALPHA 变量（写 ALPHA=1 → 编译失败 → 材质无效 → 角色隐身）；
   (c) 替换光照方程后 **BASE_COLOR 不自动参与**，基色须自己乘进 DIFFUSE 累积。
6. **调试方法论教训**：截屏曾被外部 screencapture（无 TCC 权限静默失败）卡住看旧图、
   曾把壁纸岩纹误读为"隐身角色"——**判定渲染问题先做像素级指纹测试（改纯色底 +
   像素统计），别用肉眼对比相似截图**。
7. Hunyuan 全网格本身 ≈3 万表面碎片瓦片（渲染实心）；任何减面器把它打成单面碎片
   （雪花/隐身）。连通减面 = pymeshlab 焊接+decimate 管线（后续优化项，M2 后）。

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
