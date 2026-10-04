# 交互音效资产（v0.19.0 F4）

音效管线默认关（config `sound.enabled`）。缺资产时静默降级，随时可补、
不算破坏性变更。

命名约定（`pet/sound.py` 的 `SoundFX` 按 `<name>.wav` 查找本目录）：

| 文件 | 触发 |
|---|---|
| `pet.wav` | 摸摸头（单击） |
| `feed.wav` | 喂食（双击/菜单；咀嚼短音） |
| `clean.wav` | 洗澡/梳毛 |
| `poke.wav` | 戳一戳（0.19.1 起改"逗一逗"） |
| `reject.wav` | 饱和拒绝（0.19.1 F6 启用） |

建议规格：44.1kHz 单声道 wav、0.2–0.6s；整体音量已由 config
`sound.volume`（0–1）统一控制，素材按正常响度制作即可。
