"""Blender 开工场景生成器（M2 对位建模第一步）。

用法：
  blender --background --python three_d/tools/build_starter_blend.py -- \
      <skeleton3d_adult.json> <out.blend> <draft_hunyuan.glb>

产出：
  - 双正交图板（front/front_rest 为正视参考、side/side_key 为侧视参考），脚底贴 z=0；
  - 47 骨基准骨架（关节=bone head，米制，y↔z 轴互换：json y-up → Blender z-up）；
  - 混元草模（已 1.5m/落地/Y-up，glTF 导入器自动转 Z-up）；
  - 场景单位=米。
"""
import json
import math
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
json_path, out_path, draft_glb = argv[0], argv[1], argv[2]
REPO = "/Users/zzh4206/Desktop_Pet"

sk = json.load(open(json_path))
ppm = float(sk["meta"]["px_per_meter_proposal"])
W, H = sk["meta"]["canvas_px"]
bones = sk["bones"]

bpy.ops.wm.read_factory_settings(use_empty=True)
scn = bpy.context.scene
scn.unit_settings.system = "METRIC"
scn.unit_settings.scale_length = 1.0

BOARD_H = H / ppm          # 图板实际高度 ≈1.64m
board_front = f"{REPO}/assets/rig_adult_walk_v1/references/front_rest.png"
board_side = f"{REPO}/assets/rig_adult_walk_v1/references/side_key.png"


def add_board(path, name, loc, rot_y):
    img = bpy.data.images.load(path)
    e = bpy.data.objects.new(name, None)
    e.empty_display_type = "IMAGE"
    e.data = img
    e.empty_display_size = BOARD_H
    e.location = loc
    e.rotation_euler = (math.radians(90), math.radians(rot_y), 0)
    scn.collection.objects.link(e)


# 正视图板：位于 XZ 平面（正视图看到），稍微推到角色前方
add_board(board_front, "board_front_XY", (0.0, -0.03, BOARD_H / 2), 0)
# 侧视图板：旋转 90° 落到 YZ 平面（右视图看到），推到角色侧方
add_board(board_side, "board_side_ZY", (0.03, 0.0, BOARD_H / 2), -90)

# ---- 骨架 ----
pos_m = {}
for b in bones:
    x, y, z = b["pos_px"]           # json: x 右, y 上, z 面朝
    pos_m[b["name"]] = Vector((x / ppm, z / ppm, y / ppm))  # → blender x右/y前/z上

children = {}
for b in bones:
    if b["parent"]:
        children.setdefault(b["parent"], []).append(b["name"])

MIN_LEN = 0.008
arm = bpy.data.armatures.new("skeleton3d_ref")
arm_obj = bpy.data.objects.new("skeleton3d_ref", arm)
scn.collection.objects.link(arm_obj)
bpy.context.view_layer.objects.active = arm_obj
bpy.ops.object.mode_set(mode="EDIT")
eb = arm.edit_bones

src2eb = {}
for b in bones:                     # 第一遍：建骨、定 head/tail
    e = eb.new(b["name"])
    e.head = pos_m[b["name"]]
    kids = children.get(b["name"])
    tail = pos_m[kids[0]] if kids else e.head + Vector((0, 0, MIN_LEN))
    if (tail - e.head).length < MIN_LEN:
        tail = e.head + Vector((0, 0, MIN_LEN))
    e.tail = tail
    src2eb[b["name"]] = e
for b in bones:                     # 第二遍：挂父（json 顺序保证父先建）
    if b["parent"]:
        src2eb[b["name"]].parent = src2eb[b["parent"]]
    src2eb[b["name"]]["src_vrm"] = b.get("vrm") or ""
    src2eb[b["name"]]["method"] = b.get("method", "")
bpy.ops.object.mode_set(mode="OBJECT")

# ---- 草模 ----
bpy.ops.import_scene.gltf(filepath=draft_glb)
for o in bpy.context.selected_objects:
    o.name = "draft_hunyuan_" + o.name

bpy.ops.wm.save_as_mainfile(filepath=out_path)
print("SAVED:", out_path)
