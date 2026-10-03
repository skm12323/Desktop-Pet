# ADULT 视觉流程复盘与 FINAL 复用指南

日期：2026-10-01　依据：`docs/ADULT*.md` 全部记录（2026-09-25 → 10-01，v0.16.4 → v0.16.14）、`2D骨骼蒙皮动效全栈工作流规范.md`、当前代码与资产。

## 0. 结论

**FINAL 适合复用 ADULT 的“正面骨骼 ⇄ 转身片段 ⇄ 侧身骨骼”架构和大部分工具，但不能直接照搬。** 主要原因有三点：

1. **FINAL 还没有正面蒙皮骨骼。** `assets/rig/final/` 目前仍是纸娃娃包，只有 3 张整图和若干摆动件。ADULT 开始这条路线时已经有正面骨骼，FINAL 则要多做一整段正面骨骼。
2. **FINAL 原画不是静止站姿。** `assets/reference/final_ref.jpg` 中右手提裙、左手挥手、张嘴、歪头。拆层前必须先生成一张放松站姿的正面关键原画。
3. **服装结构不同，难点会转移。** FINAL 的长裙垂到脚踝，分为外层藏青开襟裙、内层蓝色条纹裙和衬裙荷叶边，裙面有大面积金色花纹；长发及腰且向两侧铺开。ADULT 的膝部问题（V05、小腿反弯、大腿穿出裙侧）在 FINAL 中大多被裙子遮住。主要风险变成**长裙随步伐的形变**，以及**长发和摆臂之间的穿插**。ADULT 的 V04 就是同类问题：裙纹拉伸修复前为 1.96×，降低跟随增益、加入弹簧后为 1.38×。ADULT 的裙子只到膝部，FINAL 的长裙会把这一问题放大。

另外，ADULT 的运行时和部分工具写死了 `adult`。开始 FINAL 资产前，应先完成一次代码泛化（见 §3 F0）。

---

## 1. ADULT 的实际经过（时间线）

| 日期 | 阶段 | 主要事件 | 文档 |
|---|---|---|---|
| 09-25 | 审查 | 静态站姿可用；正面行走无法通过调参修好，因为腿在正面画面中只能外展和交叉。完成贴地与放松站姿修复 | 视觉审查与自然行走方案 |
| 09-25 | 弯路 | GLM“连续视角关键形态”方案的 28/28 数值门禁全部通过，但画面只是正面图被横向压扁 → **停止该路线**，并确立“数值 + 目检双签”规则 | 连续视角执行计划、步态计算核实现记录 |
| 09-27 | 新方案 | 正面骨骼 ⇄ 视频转身 ⇄ 侧身骨骼；G0 基线（开启 mipmap）、G1 侧身原画（gpt-image-2.5 + Qwen 姿态参考）、G2 本地 Wan 2.2 打通流程、确定取景 | 转身视频过渡与侧身骨骼行走方案 §9 |
| 09-28 | G3–G6 | 抠像与后处理、SAM 切分侧身骨骼、gait 接入、状态机与单时钟；49/49 | 同上 |
| 09-28 | 步态复查 ×3 | 膝盖反折 → 腿根横移 → 小腿假关节（root_lock）→ 停步交叉 → 腿偏前、手臂僵硬 | 侧身步态复查 |
| 09-29 | 结构修复 | 诊断出四类问题：部件涂抹、权重过散、步态不像人、扭曲。按 A 权重 → 分段刚体手臂 → 远侧手臂整条重绘 → C 人体参考曲线的顺序处理 | 行走修复 |
| 09-30 | 视觉审查 | V01–V07 问题报告 → 首轮修复（各情绪行走、腋下补全）→ 第二轮修复（alpha≥240、色边、裙摆、膝部曲线、尾鳍、反向等待）；修复眨眼白眼网格 bug | 视觉问题报告 / 首轮 / 第二轮 / 眨眼 |
| 10-01 | 转身片段 | Seedance 被否、Wan 2.2 720p 出现重影，最终采用 Wan 3.0 API 1080P；新增 1024 高帧包 | 转身片段换 Wan3 |

### 1.1 当前状态（2026-10-02 按 `62b3342` / v0.16.14 核对）

本节由代码、资产和测试实测得到：

- **运行时**：`adult_locomotion` 的默认值、`config.example.json` 和本机 `config.json` 均为 `side_rig`。转身帧包有 256/512/1024 三档（各 37 帧 @30 fps，约 1.2 s，取自 Wan3 片段 0–78 帧），六个包都已经过 `polish_turn_frames`。片段播放速度为普通 1.25×、反向 2×，反向归位 0.12 s。neglected 用 MultiEffect，`saturation -0.65`。
- **测试**（全部用 Python 3.12 + D3D11 运行）：转身/行走场景 39/39、情绪 68/68、成年体骨骼 36/36、第二轮视觉 8/8、round2 8/8、自然度通过、旧步态 28/28、眨眼 3/3。
- **尚未完成**：
  - 真机内存浸泡测试（旧记录稳态约 205 MB，加侧身行走约 +30 MB，红线 200 MB）；
  - 真机确认转身中脚底和尾鳍的表现；
  - 512 档帧包解码后 14.6 MB，超过 12 MB 的单条门禁；
  - 多 DPI 和软件渲染回退没有测过（09-30 审查的范围说明）；
  - 文档中没有用户对 Wan3 版本的书面签字记录（5.3 最终验收）。

**返工原因归类**：生成部件不干净（程序补全）、权重算法沿用 YOUNG 的做法、只验数值不看画面、只验 `healthy_neutral`、只看静止画面没看运动暴露区。下文 §2 的经验基本对应这几类。

---

## 2. 经验（按环节）

### 2.1 验收方法

1. **数值门禁和目检必须双签。** 数值全过但目检不过，仍判为不通过。用户目检 GIF / 高清参考 GIF 是最终依据（`tools/render_adult_reference_gif.py`）。
2. **在真实 256 窗口测运动学。** 步态按桌面像素速度运行，窗口越大，相对步幅越小；640 窗口测出的数据已作废。
3. **接触和漂移指标一律取蒙皮后的网格。** 必须经过完整显示变换（fit、groundShift、镜像、窗口），不能拿骨骼标记代替。
4. **静止复原好，不代表部件完整。** 运动会暴露遮挡区（V01）。必须做极限姿态检查，并按材料边界区分“缺口”和“正常留白”（09-30 第二轮新增的分类检查）。
5. **“无翻折”不代表形变合格。** 要看纹理有效区的局部拉伸和压缩 σmax/σmin（`spikes/qa_deform_metrics.py`、visible_deformation）。
6. **每个情绪和分支都要验收。** 只测 `healthy_neutral` 时，漏掉了其他状态行走退回旧两帧的问题（V02）。
7. **契约测试要跟设计一起更新**，例如加入 `walk_park` 状态、腿根覆盖断言。数值阈值不能为了通过而放宽；修改判据定义时要写明理由并请用户确认。

### 2.2 原画与生成

1. **参考图优先，文字只做补充。** 饰品换边（蝴蝶结跑到近侧）靠文字压不住，要先修正参考图。
2. **一次只跑一个生成任务**，期间不并行其他重负载。ComfyUI 按任务启动，用完关闭。
3. **单独部件用 #00B140 绿底生成**，因为白底会误伤白色荷叶边。
4. **大部分被遮挡的肢体或部件要整条重画。** 不能把原画可见窄条和生成图拼接，错位约 10 px 就会出现双层袖口。邻色扩散、外推之类的程序补全看起来像“粘”在遮挡物边上，已全部被替换。
5. **Qwen 的几个坑**：双参考编辑会把背景画成灰噪点，要改用 BiRefNet 抠像，不要再“换白底”；“去掉 X”要对整块做无遮罩编辑，带遮罩采样反而会把 X 画回来；“只重画部件 X”配合 BiRefNet 能得到干净部件。
6. **gpt-image 的输出 alpha 只有 253**，需要按 ×255/253 修正。内部遮挡判定用 alpha ≥240，原图中有 252–254 的像素会被误丢。

### 2.3 拆层与蒙皮

1. **用 SAM 3.1 按像素切分原画，只补遮挡区**，静止合成可以与原画逐像素一致。文字提示比点提示可靠；See-Through 只适合当补全来源。
2. **左右按画面侧命名**：`_l` 是画面左侧，即角色自身右侧。侧身骨骼沿用同名骨骼表示同一身体部位；正面 47 骨，侧身取其中 37 骨，22 层。
3. **YOUNG 的反距离权重不适合长肢体。** 改用 `weights.mode=chain`，加内侧铰链加宽带和 `min_component_px`。2D DQS 没有改善，已撤回。
4. **手臂按分段刚体处理，关节处加圆盘盖**（`split_hinged_limb.py`）。尾鳍由尾鳍骨刚性驱动；腿用局部 Hermite 曲线 `joint_curve`。
5. **膝支点必须在髋–踝连线之前**，否则 IK 在静止时就会解成反向膝。IK 的 `exact_within` 要求 C1 连续。不要给腿加骨盆锁定 root_lock，否则会出现假关节。
6. **裙摆跟腿的增益要小**（`skirt_follow_gain` 0.8 → 0.4，`skirt_follow_limit_deg` 8，弹簧 3.5 Hz / 半衰期 0.1 s）。增益大会拉扯花纹。现有裙子只用 `skirt_hem_l/r` 两根下摆骨，靠 `weights.mode=skirt` 按高度渐入。
7. **蒙皮纹理开启 mipmap**，片段帧也走同一条 GPU 缩放路径，否则交接处会出现画质突变。
8. **追加网格边界时不能取列表末元素作上界**，这正是眨眼露白眼 bug 的根因。

### 2.4 转身片段

1. **首尾帧用骨骼的实际静止渲染，不用原画**；拆层不是逐像素复制。
2. **模型选择**：Wan 3.0 API（`tools/wan3_flf2v_api.py`，1080P，`watermark=false`，key 只放环境变量）能做首尾帧，动作自然。Seedance 会整体刚性旋转，Wan 2.2 加 4 步 LoRA 到 720p 会重影，超分补不出细节，这三种都被否了。（注：09-27 方案 §4.1 写“Wan 3.0 只支持首帧”，后经实测更正。）
3. **按尾巴甩动方向设置不对称取景**（ADULT 为左 160 / 右 40）。本地 Wan 输出 480×848，不是严格 9:16，x、y 缩放要分开算。
4. **后处理**：只在静止保持帧上做端点形变；在运动帧上溶解会出现尾巴重影。按运动重定时，再加两段式交叉淡化。用 `polish_turn_frames.py` 清理青绿色边。
5. **转回正面目前倒放同一条 Wan3 片段**（`--reverse`，`clip.json` 中 `reversed: true`）。这推翻了 09-27 方案中“转回单独生成，不用倒放”的原则，因为倒放时头发会先于身体动。倒放效果只逐帧检查了关键帧，中段的脚底滑动和尾鳍稳定性还要在真机上确认。FINAL 的长发甩动幅度更大，应先比较倒放和单独生成两种做法。
6. **按窗口高度提供 256 / 512 / 1024 三档帧包。**

### 2.5 步态与运行时

1. **不用生成式行走循环**，行走只走骨骼 + IK。
2. **步态参数参照人体矢状面曲线**（`reference_curves`）：着地前伸直膝、承重时屈膝缓冲、摆动期膝角单峰、手臂与同侧腿反相、肘部始终微屈。ADULT 目前只是接近参考值：着地膝约 17°、支撑中期约 10°，参考值为 0–5°，受膝支点前置和 IK 伸直奇异的限制。当前参数为 1.2 Hz、120 px/s，上限 200 px/s。
3. **停步（WALK_PARK）只许向前收脚**，不许后退或交叉，也不能停在踮脚状态。
4. **单时钟**：行走期间窗口 x 和姿态在同一 60 Hz 拍提交；行为层只给出意图，位置从窗口回读。
5. **neglected 分支用 MultiEffect 去饱和**，整个会话从头到尾保持灰暗；行走期间用中性脸。
6. **控制反向的等待时间**：片段加速播放（普通 1.25×、反向 2×）。

### 2.6 工程

- 新资产放进独立目录，不碰生产包；工具都要带路径参数和 `--dry-run`。`test_skinned_mesh_adult` 会改写 preview png，跑完需要复原。
- 跑 Qt 测试要用 Python 3.12（PySide6 6.11），并设 `QT_QPA_PLATFORM=windows QT_QUICK_BACKEND=rhi QSG_RHI_BACKEND=d3d11`。该解释器已于 2026-10-02 装好 scipy 1.18.1（numpy 2.5.2 未变），可以跑全部 ADULT 测试；不要再用 anaconda 跑 Qt 测试，它缺 Qt 硬件后端。测试会改写 `assets/rig_adult/preview_skinned_adult_live.png`、`spikes/_qa/adult_blink_return_2026-09-30/*`、`spikes/_qa/adult_visual_round2_2026-09-30/fixture_*.png`，跑完用 `git checkout` 复原。
- 内存红线是 200 MB，侧身骨骼加片段约 +30 MB，仍需真机实测。

---

## 3. FINAL 复用流程（建议）

| 步骤 | 内容 | 复用 / 新做 | 门禁 |
|---|---|---|---|
| **F0 代码泛化** | `presenter.enable_side_locomotion` / `set_mood_locomotion` 的 `stage != "adult"` 判断、`app.py` 写死的 `rig_adult_walk_v1` 路径和 `adult_locomotion` 配置改成按阶段配置；`build_side_spec.py`（无 argparse，骨骼坐标、图层配置、PKG 全部写死）改为读取 JSON；`render_adult_reference_gif.py` 加 `--stage`。`render_rig_rest.py` 已支持 `--stage final` 和 `--rig-dir`，`build_side_layers.py`、`side_completions.py`、`prepare_video_endpoints.py`、`process_turn_clip.py` 已有路径参数，只需改默认值或传参 | 改造 | ADULT 全部套件保持全绿 |
| **F1 正面关键原画** | gpt-image-2.5：图 1 = `final_ref.jpg`（身份），图 2 = ADULT 正面静止渲染或 `adult_ref.png`（放松站姿、构图）。要求双手自然下垂、闭口、正头，鞋面露出裙底；对齐画布和脚底线 | 新做 | G1 同类判据 + 目检 |
| **F2 正面骨骼** | SAM 切分 + Qwen/GPT 整件重画遮挡区（绿底）→ 47 骨 spec → 链式权重网格 → `rest_pose_angles`、`ground_anchor_y_px`、`texture_mipmaps` → 眨眼网格检查 | 工具复用，资产新做 | G4 静止复原 / 极限姿态 / 权重；眨眼测试 |
| **F3 侧身关键原画** | 姿态参考可用 `assets/rig/final/figs/healthy_side.png`（角度和发型合适，但正在迈步、双脚藏在裙下）或 ADULT `side_key.png`；要求双脚平放、鞋尖露出 | 新做 | G1 |
| **F4 侧身骨骼** | 沿用 SAM → `build_side_layers` → `side_completions` → 分段手臂 → mesh，**新增长裙分层**（见 §4） | 工具复用 + 裙子新做 | G4 + 材料缺口分类 + 有效区形变 |
| **F5 转身片段** | `prepare_video_endpoints`（按 FINAL 的发型和尾巴重新试取景）→ Wan 3.0 API → `process_turn_clip` → `polish_turn_frames` → 三档帧包 | 直接复用 | G2/G3 + 接缝 ≤3/255 |
| **F6 步态** | 复用 gait、参考曲线、walk_park；按长裙重新调步幅和抬脚（裙摆限制步幅，适合小步） | 参数新调 | G5 + round2 + naturalness |
| **F7 集成与情绪** | side_locomotion 状态机、片段图层、neglected 去饱和；FINAL 十个 figure 逐个验收 | 复用 | G6 场景 + moods 套件 |
| **F8 视觉审查轮** | 照 ADULT 09-30 的方法出 V 编号问题报告 → 分轮修复 → 高清参考 GIF 给用户签字 | 复用方法 | 双签 |

建议的执行顺序：**F0 → 先做长裙原型（§4）→ F1 → F2 → F3 → F4 → F5 → F6 → F7 → F8**。长裙是最大的未知风险，应先用 ADULT 的侧身骨骼或 FINAL 现有的 `healthy_side.png` 做一个低成本原型，确定变形方案后再投入正式拆层。

---

## 4. FINAL 特有风险与对策

| 风险 | 说明 | 对策 |
|---|---|---|
| 长裙形变 | 裙子是行走中面积最大的运动部件。ADULT 只用 2 根下摆骨跟随大腿，修复前花纹拉伸 1.96×，压低增益后为 1.38×，代价是裙摆跟随幅度只剩 ±8°；长裙不能靠同样的办法压低 | 按外层开襟裙前后片、内层条纹裙、衬裙荷叶边分层；腰带刚性；下摆用多骨或笼形（MVC）变形，再加弹簧滞后；用有效区 σ 和花纹宽度做门禁 |
| 腿与裙摆的关系 | 前腿迈步时裙前缘应被顶起，否则看起来像脚在“裙下平移” | 裙前缘对近侧小腿或脚加接触驱动（小增益），并设抬脚上限，避免鞋从裙中穿出 |
| 长发 | 后发面积大，摆臂时会和头发前后穿插；转身时头发甩动幅度大 | 远侧手臂放在后发之后、近侧手臂在前，层序固定；发链加长、增加骨节；视频取景按头发宽度留边 |
| 转身取景 | ADULT 的左 160 / 右 40 是按尾巴确定的。FINAL 的头发向两侧铺开，尾巴也更大 | 先用本地 Wan 2.2 零成本试取景（检查“不出画”门禁），再调用 Wan 3.0 API |
| 原画不是站姿 | 拆层要求双手下垂、双脚可见 | F1 先生成；身份参考用原画，姿态参考用 ADULT 渲染 |
| 工作量 | ADULT 仅侧身加转身就用了约 7 天、十余轮返工；FINAL 还要加上正面骨骼 | 先做 F0 和长裙原型，避免返工集中在后期 |
| 内存 | FINAL 同时只加载一个阶段，与 ADULT 不叠加；但 1024 帧包解码约 57 MB | 仍按窗口高度选档，并真机测 30 分钟浸泡 |

---

## 5. 关键文件索引

- 方案与门禁：`docs/ADULT转身视频过渡与侧身骨骼行走方案-2026-09-27.md`（§5 门禁、附录 A 提示词）
- 生成：`tools/qwen_edit.py`、`tools/birefnet_matte.py`、`tools/sam3_segment.py`、`tools/wan_flf2v.py`、`tools/wan3_flf2v_api.py`、`tools/upscale_frames.py`
- 拆层与骨骼：`tools/render_rig_rest.py`、`tools/align_side_key.py`、`tools/build_side_layers.py`、`tools/side_completions.py`、`tools/split_hinged_limb.py`、`tools/build_side_spec.py`、`tools/mesh_generator.py`
- 片段：`tools/prepare_video_endpoints.py`、`tools/process_turn_clip.py`、`tools/polish_turn_frames.py`
- 运行时：`pet/rig/gait.py`、`pet/rig/side_locomotion.py`、`pet/rig/presenter.py`、`pet/rig/rig_scene.qml`、`pet/rig/skinned_mesh_item.py`、`app.py`
- QA 与测试：`spikes/qa_side_rig.py`、`spikes/qa_turn_clip.py`、`spikes/qa_deform_metrics.py`、`spikes/qa_adult_visual_report.py`、`spikes/test_adult_locomotion*.py`、`spikes/test_side_gait_*.py`、`spikes/test_adult_visual_round2.py`、`spikes/test_adult_blink_return.py`
- 交付 GIF：`tools/render_adult_reference_gif.py`
