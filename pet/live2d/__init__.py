"""Live2D Cubism 展示后端 —— 与 frames/rig/paperdoll 并列的第四档。

运行时走 ``live2d-py``（Cubism Native 的 Python 绑定），模型为标准
``.model3.json``。缺库、缺模型、OpenGL 初始化失败一律回退 frames，
展示层永不阻断启动。
"""

from .spec import Live2DMapping, load_live2d_mapping, resolve_model_path

__all__ = ["Live2DMapping", "load_live2d_mapping", "resolve_model_path"]
