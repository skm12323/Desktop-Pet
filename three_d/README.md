# three_d/ — 三维化实验线（独立管理）

> 2026-10-03 立项。本文件夹单独管理 3D 线的全部资料：wiki、调研结论、DCC 资产源文件。
> 运行时代码**不在这里**——进 `pet/render3d/`（跟随包结构、可测试、随安装包分发）。

## 目录

```
three_d/
├── wiki/          # 本线专属 wiki（导航见 wiki/总览.md）
├── assets_src/    # DCC 源文件（.blend / VRM / 贴图 PSD / 图板）——不入库（.gitignore）
└── README.md      # 本文件
```

## 与主仓库的关系

- 主 2D 线（`pet/rig/`、`assets/rig_*/`）继续维护不动，是默认呈现。
- 3D 是实验轨：config `render3d.enabled` 默认 `false`，云端推送默认关闭。
- 降级矩阵不变：3D → 2D rig → frames，任何异常静默退下一级。
- 决策与调研全部记录在 `wiki/`，主 `wiki/` 不重复记录本线内容。

## 快速入口

- 立项与全部决策：[wiki/方案-启动与决策记录.md](wiki/方案-启动与决策记录.md)
- 导航：[wiki/总览.md](wiki/总览.md)
