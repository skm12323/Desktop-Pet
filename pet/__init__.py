"""桌宠核心包。"""

# 版本单一源：app.APP_VERSION 由本值派生（发版只改这里）。
# v0.16.4–0.16.6：成年转身与步态 G0–G4 合并入档 / ADULT 侧身行走 G5+G6
# （side_rig 默认启用）/ QSGTexture 释放改 deleteLater（G6 双骨骼退出段错误）。
# v0.16.7：rig 渲染循环三档自适应降频（16/33/66ms，与 side 编排合流）+
# 长运行 GC 治理（gc.freeze + 阈值放宽）——idle 稳态 CPU −35%
# （19.4%→12.6% / 17min 浸泡 27.2%→17.3%），内存持平。
__version__ = "0.16.7"
