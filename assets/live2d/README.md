# Live2D 模型资产

本目录放 Cubism 3+ 模型（`.model3.json` + `.moc3` + 贴图/动作/表情）。

默认样例是官方 **Haru**（`haru/`），来自 [CubismWebSamples](https://github.com/Live2D/CubismWebSamples)，适用 [Live2D Free Material License](https://www.live2d.com/eula/live2d-free-material-license-agreement_en.html)。完整条款见 [`CUBISM-SAMPLES-LICENSE.md`](CUBISM-SAMPLES-LICENSE.md)。年营收超过 1000 万日元的商业使用还需 Cubism SDK 发布许可。

## 换成自己的模型

1. 把模型目录拷进 `assets/live2d/<name>/`
2. 在同目录放一份 `mapping.json`（可复制 `haru/mapping.json`），把 `expressions` / `motions` 改成该模型真实的表情名和动作组
3. `config.json`：

```json
{
  "presentation": "live2d",
  "live2d": {
    "model": "assets/live2d/<name>/<File>.model3.json"
  }
}
```

缺运行时（`live2d-py`）或缺模型文件时自动回退 `frames`，不阻断启动。
