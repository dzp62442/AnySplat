# AnySplat：OmniScene 数据集实验文档

本文记录已确认的实验方案、代码实现与实验结果。两套配置、训练、自动续训、mini/total 评估和作者权重评估均已接入。第 1～11 节保留首轮方案与实现依据；第 12 节记录 2026-10-03 手动停止的首轮 112×200 实验及分析；第 13 节记录用户确认的已知输入相机适配方案；第 14 节记录其实现与检查。首轮未完成 100001 步，也未执行完整 total 评估；没有审查或重新预处理数据集。

初始方案核查日期：2026-10-02；实验记录更新日期：2026-10-03（北京时间）。初始代码依据如下，首轮适配代码提交及运行版本另见第 12 节：

| 项目 | 分支 | 核查提交 |
| --- | --- | --- |
| AnySplat | `comp_svfgs`，与作者 `main` 相同 | `9f1c4aeccc3b96947478be584fdb9aebe94512d5` |
| SVF-GS | `main` | `d8d794836ef5eb76c0e659f83310dfea21839ec6` |
| depthsplat | `comp_svfgs` | `265ebdea10d7b18ad461bfe9520736e496c5874f` |

## 1. 已确认的实验边界与选择

### 1.1 输入、监督与评估

重建对象是中心时刻的场景。每个 bin 仅将这一时刻的 6 路环视 RGB 输入 AnySplat，一次前馈生成该时刻的高斯。前后时刻的 12 路图像只作新视角评估真值，不进入重建网络，不参加训练损失；不使用历史帧特征、跨时刻状态或多帧高斯融合。

| 信息 | 重建网络输入 | 训练监督 | 评估用途 |
| --- | --- | --- | --- |
| 中心时刻 6 路 RGB | 是，唯一外部输入 | 输入视角 RGB 重建损失 | `input_6`，并包含在 `all_18` 中 |
| 6 路输入的已知内外参 | 否 | 不作为相机真值监督 | 仅用于评估坐标对齐及渲染 |
| 前后 12 路 RGB | 否 | 否 | `novel_12` 真值 |
| 18 路目标相机内外参 | 否 | 否 | 指定渲染视角 |
| Metric3D 尺度深度及置信度 | 否 | 否 | 本方案不用，不建立读取依赖 |
| 动态物体掩码 | 否 | 保留 SVF-GS 的使用约定，见下文 | 正式四项指标不乘训练掩码 |
| DepthAnything V2 相对深度 | 否 | 否 | 仅计算 PCC |
| 网络自行预测的深度、相机与 VGGT 伪几何 | 网络内部产生 | 沿用 AnySplat 选定配置 | 不视为额外外部输入 |

不获取 LiDAR 点云、天空掩码等额外信息，也不离线补充几何标签。已有相机外参以 LiDAR 坐标系命名，并不意味着使用 LiDAR 测量数据；可以使用这些已提供的相机变换，但加载器不依赖 `LIDAR_TOP` 条目或点云文件。

直接相信 OmniScene 的数据契约，不增加数据质量预扫描、有效性筛选、场景黑名单、位姿或深度阈值审核，不因自行判断数据“不合理”中断或跳过样本。不重新选帧、筛 bin、重建划分。正常文件读取和张量计算之外，不引入数据审查流程。

### 1.2 已与用户确认的决定

| 事项 | 决定 |
| --- | --- |
| 原作者配置基准 | **`multi-dataset.yaml`**；保留其模型、优化器与损失设置，训练数据替换为单一 OmniScene |
| 两个分辨率的训练初始化 | 沿用作者训练入口：VGGT 预训练初始化，训练 AnySplat；不默认加载已训练好的 AnySplat 微调 |
| 动态掩码 | **严格沿用 SVF-GS：输入 6 视角使用全 1 掩码** |
| 尺寸不可整除 | 使用 DepthSplat 的**中心裁剪到 patch 整数倍**策略，适配 AnySplat 的 patch size 14 |
| 评估坐标系 | 仅用 6 路输入的已知与预测相机，估计一个全局相似变换 |
| 作者模型 | 单独支持官方 `lhjiang/anysplat` 权重直接评估，两个分辨率均支持 |

AnySplat 当前没有 RE10K 实验配置，也没有 DepthSplat 那样的 Small/Base/Large 成品选项，因此不人为创建一个所谓“AnySplat Base”。采用本仓库已有 AnySplat 架构与作者发布模型。

### 1.3 为什么选 multi-dataset；深度蒸馏是什么

DepthSplat 的 OmniScene 配置来自其 RE10K 实验，显式损失为 MSE 与 LPIPS，不使用外部深度真值监督。AnySplat 的 DL3DV 与 multi-dataset 配置大体相同，但直接深度蒸馏权重分别为 `1.0` 和 `0.0`。因此在这一项上，multi-dataset 更接近 DepthSplat；两者并非完全相同的训练方法。

AnySplat 的“深度蒸馏”是：把同一批输入 RGB 送入冻结的 VGGT，在线预测一份深度，用它约束高斯渲染深度。它不是读取 Metric3D，也不是使用新视角真值。选择 multi-dataset 后，此项权重为 0，但仍保留：

1. 位姿蒸馏：学生预测相机对齐冻结 VGGT 的预测相机。
2. 深度一致性：高斯渲染深度与学生深度头输出保持一致，使用 VGGT 置信度选择内部监督区域。

所以 `weight_depth: 0.0` 不等于 `distill: false`。本方案保留 `distill: true`，不能为模仿 DepthSplat 而删去 AnySplat 原有几何训练机制。

## 2. 配置组织与实验入口

### 2.1 配置文件

| 文件 | 作用 |
| --- | --- |
| `config/dataset/omniscene.yaml` | 根目录、版本、划分、固定 6 路输入、加载尺寸、掩码及 PCC 深度读取策略 |
| `config/experiment/omniscene_112x200.yaml` | 112×200 加载分辨率的完整实验 |
| `config/experiment/omniscene_224x400.yaml` | 224×400 加载分辨率的完整实验 |

沿用 AnySplat 的 Hydra 入口 `python -m src.main +experiment=...`。保留它的 `dataset.omniscene` 包装结构，并在 `src/dataset/__init__.py`、`src/config.py` 以及对应 dataclass 中注册配置；不能直接复制 DepthSplat 的单层 `dataset.image_shape` 结构。

两套实验共享模型与训练参数，只改变数据加载分辨率、实验名和工作目录。配置从 multi-dataset 提取继承项，但不继承其中 DL3DV、CO3D、ScanNet++ 三个数据源或两节点训练设置。

下列为配置契约。“新增”指相对于作者 main 增加的字段，已经连同 typed config 和调用逻辑实现。完整配置见 config/experiment/omniscene_common.yaml 及两套分辨率配置。

```yaml
# 共同设置摘要；运行时加载仓库中的 experiment 配置。
defaults:
  - /dataset@_group_.omniscene: omniscene
  - override /model/encoder: anysplat
  - override /model/encoder/backbone: croco
  - override /loss: [mse, lpips, depth_consis]

wandb:
  mode: offline

model:
  encoder:
    gs_params_head_type: dpt_gs
    pose_free: true
    pretrained_weights: ''
    pred_pose: true
    pred_head_type: depth
    freeze_backbone: false
    freeze_module: patch_embed
    distill: true
    voxelize: true
    voxel_size: 0.002
    anchor_feat_dim: 128
    gs_prune: false
    render_conf: false
    conf_threshold: 0.1
    intermediate_layer_idx: [4, 11, 17, 23]
    gaussians_per_pixel: 1
    num_surfaces: 1
    gaussian_adapter:
      gaussian_scale_min: 0.5
      gaussian_scale_max: 15.0
      sh_degree: 4
  decoder:
    name: splatting_cuda
    background_color: [1.0, 1.0, 1.0]
    make_scale_invariant: false

loss:
  mse: {weight: 1.0, conf: false}
  lpips: {weight: 0.05, apply_after_step: 0, conf: false}
  depth_consis: {weight: 0.1, loss_type: MSE}

dataset:
  omniscene:
    roots: [datasets/omniscene]
    data_version: interp_12Hz_trainval
    input_image_shape: [112, 200]  # 另一套为 [224, 400]
    num_context_views: 6          # 新增：固定中心时刻 6 路
    load_metric_depth: false     # 新增
    load_dynamic_mask: true      # 新增：遵循输入全 1、目标真实掩码约定

data_loader:
  train: {batch_size: 1, num_workers: 16, persistent_workers: true, seed: 1234}
  val: {batch_size: 1, num_workers: 1, persistent_workers: true, seed: 3456}
  test: {batch_size: 1, num_workers: 4, persistent_workers: false, seed: 2345}

trainer:
  max_steps: 100_001
  val_check_interval: 1000
  num_nodes: 1
  devices: 1                    # 新增；避免当前 devices="auto" 隐式多卡
  accumulate_grad_batches: 1
  precision: bf16-mixed
  gradient_clip_val: 0.5
  num_sanity_val_steps: 0        # 新增；不将启动检查计入验证次数

train:
  supervision_views: context    # 新增
  use_dynamic_mask: true        # 新增
  eval_model_every_n_val: 10    # 新增
  final_mini_test: true         # 新增
  pose_loss_alpha: 1.0
  pose_loss_delta: 1.0
  cxt_depth_weight: 0.0
  weight_pose: 10.0
  weight_depth: 0.0
  weight_normal: 0.0

optimizer:
  lr: 2.0e-4
  warm_up_steps: 1000
  backbone_lr_multiplier: 0.1

checkpointing:
  load: null
  auto_resume: true             # 新增
  init_source: vggt             # 新增；不是完整断点续训
  every_n_train_steps: 200
  save_top_k: 5
  save_weights_only: false
  save_last: true               # 新增

test:
  split: total                 # 新增；训练中评估固定使用 mini
  weights_source: training     # 新增；另一选项为 author
  pretrained_path: pretrained  # 新增；作者权重本地目录
  align_pose: false            # 关闭原来的逐视角测试优化
  camera_alignment: input_sim3 # 新增；区别于原 align_pose
  view_groups: [all_18, novel_12, input_6] # 新增
  compute_scores: true
  compute_pcc: true             # 新增
  eval_time_skip_steps: 5       # 新增；只排除计时预热，不排除指标样本

notifications:                  # 新增
  feishu_enabled: true
  library_root: ${oc.env:HOME}/Libraries

seed: 111123
hydra:
  run:
    dir: work_dirs/omniscene_112x200
```

`work_dirs/omniscene_224x400` 用于另一套训练。禁止默认时间戳目录造成自动续训找不到旧实验，也不跨分辨率搜索和误加载断点。新增顶层 `notifications` 需要加入 `RootCfg`。

默认单卡、单节点、无梯度累积，使全局 batch size 确实为 1。6 个视角属于同一个样本，不是 batch size 6。后续若改多卡，应单独确认全局 batch 变化；本方案不自动扩卡。

### 2.2 模型与优化器：沿用 AnySplat

| 项目 | 设置与实际含义 |
| --- | --- |
| Encoder | `anysplat`，实际几何骨干为 VGGT；`backbone: croco` 是现有配置组织，不能据此换成 DepthSplat 骨干 |
| 头部 | `gs_params_head_type: dpt_gs`，`pred_head_type: depth`，`pred_pose: true` |
| 输入几何 | `pose_free: true`；不启用外部内参、位姿或深度条件输入 |
| 骨干初始化 | `facebook/VGGT-1B`；高斯相关新增层按作者构造逻辑初始化 |
| 冻结策略 | `freeze_backbone: false`，`freeze_module: patch_embed`；教师模块冻结 |
| 内部蒸馏 | `distill: true`，损失权重采用 multi-dataset |
| 体素化 | `voxelize: true`，`voxel_size: 0.002`，`anchor_feat_dim: 128` |
| 高斯 | `gaussians_per_pixel: 1`，`num_surfaces: 1`，`sh_degree: 4` |
| 高斯尺度配置 | `gaussian_scale_min: 0.5`，`gaussian_scale_max: 15.0` |
| 其他继承项 | `gs_prune: false`，`render_conf: false`，`conf_threshold: 0.1`，`intermediate_layer_idx: [4, 11, 17, 23]` |
| Decoder | `splatting_cuda`，gsplat，白色背景 `[1,1,1]`，`make_scale_invariant: false` |
| 优化器 | AdamW，`betas=(0.9,0.95)`，`weight_decay=0.05` |
| 参数组学习率 | 名称含 `gaussian_param_head` 或 `interm` 的可训练参数为 `2e-4`；其他可训练参数为 `2e-5`，以实际分组代码为准 |
| 调度 | 原有 LinearLR 预热 1000 步，再接 CosineAnnealingLR；`T_max=100001`，`eta_min=2e-5` |

保留原调度器对参数组共享 `eta_min` 的行为，不改成 DepthSplat 的单目深度学习率设置。AnySplat 当前 decoder 实际使用 `render_mode="RGB+D"`、`rasterize_mode="classic"`、`radius_clip=0.1`，并写死 `near_plane=1e-10`；接口中的 near/far 当前未传给 rasterizer。不能把配置中填写 near/far 宣称为已生效的裁剪设置。

### 2.3 损失与监督真值

训练时仅渲染 6 个**网络预测的输入相机**，与对应输入 RGB 比较；不将数据集相机替换为重建网络条件，也不将 18 视角目标用于训练。

| 损失 | 权重 | 对比对象 |
| --- | --- | --- |
| RGB MSE | `1.0` | 6 路输入 RGB 与对应渲染 RGB |
| LPIPS（VGG） | `0.05`，第 0 步启用 | 同上 |
| 深度一致性 `depth_consis` | `0.1`，MSE | 渲染深度与学生深度头输出；原有 detach 默认不启用 |
| 位姿蒸馏 | `10.0` | 学生与冻结 VGGT 的相机编码；Huber，迭代预测权重衰减 `gamma=0.6` |
| 直接深度蒸馏 | `0.0` | 渲染深度与冻结 VGGT 深度，本实验不贡献损失 |
| 法线蒸馏 | `0.0` | 本实验不贡献损失 |
| 外部输入深度监督 | `cxt_depth_weight=0.0` | 不读取或监督 Metric3D |

深度一致性仍使用原有 VGGT 深度置信度的 0.3 分位阈值，即大致保留较高置信度的 70% 像素；它是内部处理。不能将“输入动态掩码全 1”写入现有 `context.valid_mask`，因为当前损失会据此覆盖教师置信度掩码。应独立保存 `context.masks`，保留原几何损失的置信度选择逻辑。

当前蒸馏实现即使法线权重为 0 仍会读取 `context.intrinsics` 计算法线。适配时对零权重法线分支直接不计算，避免为无效项引入数据集 K 依赖；有效损失保持不变。

另一个原作者行为需要如实保留和记录：`training_step` 在 `global_step > 1000` 且总损失大于 `0.2` 时返回 `loss * 1e-10`。这是已有的梯度抑制逻辑，不是新增的数据质量审核，也不会终止程序。本方案按“训练设置以本项目为准”沿用，日志记录触发次数；100001 指优化器步骤计数，不能据此保证每一步都有显著梯度。若将来要去掉，应作为显式训练策略变更讨论。

## 3. 数据集加载及与 DepthSplat 的差异

### 3.1 路径与划分

复用现有 OmniScene 数据，不重新生成。根目录通过 `dataset.omniscene.roots` 指定，默认项目内 `datasets/omniscene`，可以由用户已有挂载或软链接提供。

```text
<root>/interp_12Hz_trainval/bins_train_3.2m.json
<root>/interp_12Hz_trainval/bins_val_3.2m.json
<root>/interp_12Hz_trainval/bin_infos_3.2m/<bin_token>.pkl
<root>/samples_small/...jpg             或 sweeps_small
<root>/samples_param_small/...json      或 sweeps_param_small
<root>/samples_mask_small/...png        或 sweeps_mask_small
<root>/samples_dpt_small/...npy         或 sweeps_dpt_small
```

最后一行是 PCC 的 DepthAnything V2 结果，不是 Metric3D 的 `*_dptm_small`。保留 DepthSplat 的数据根路径重映射方式，避免绑定旧 `/datasets/nuScenes` 绝对路径。

| 阶段 | bin 列表 | 图像与附加信息 |
| --- | --- | --- |
| train | `bins_train_3.2m.json["bins"]`，训练打乱 | 只读中心 6 路 RGB；输入掩码构造全 1；无需读取前后 12 路 RGB 或 DA2 深度 |
| val | 验证列表 `[:30000:3000][:10]` | 固定 10 个 bin；输入视角损失与可视化，不反传，不加载 PCC 深度 |
| mini test | 验证列表 `[::14][:2048]` | 中心 6 路重建，完整 18 路 RGB、相机与 DA2 深度用于评估 |
| total test | 完整验证列表，不加 mini 截断 | 同上，完整覆盖 manifest 列表 |

样本数由实际列表长度记录，不把历史数量写成数据有效性断言。DepthSplat 当前代码中的 mini 截断已经注释，默认 test 读取完整列表；不能认为直接复用其测试加载器就必然得到 mini。本项目以 `test.split=mini/total` 显式切换，禁止靠修改注释切换。

### 3.2 相机顺序与 18 路定义

固定输入顺序：

```text
CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT,
CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
```

每个相机取 `sensor_info[cam][0]` 为输入。目标按相同相机顺序依次取每个相机的 `[1,2]`，得到前 12 路；最后追加中心 6 路。

| 组名 | 目标索引 | 用途 |
| --- | --- | --- |
| `all_18` | `[0:18]` | 用户要求的全部目标视角指标 |
| `novel_12` | `[0:12]` | 用户要求的新视角指标 |
| `input_6` | `[12:18]` | 补充诊断，不替代上述两组 |

前后时刻图像的存在属于评估协议，不代表模型使用时序输入。新视角也不能被混入 VGGT 教师的输入。

### 3.3 图像、相机与尺寸处理

RGB 沿用参考加载器的图像读取、resize 与 `[0,1]` 转换。参数文件对应 224×400 小图；加载为 112×200 时同步缩放像素内参，再归一化 K 的第一行为除宽、第二行为除高。

只移植 OmniScene 固定 resize/crop 流程，不继承原多数据源加载器的随机内参增强、随机宽高或基于外部点云/深度的场景归一化；输入与监督保持同一图像变换。

外参采用 OpenCV 相机约定与列向量 `c2w`。复用 DepthSplat 的 `sensor2lidar_transform`，不复制 SVF-GS 为其渲染器增加的 Y/Z 轴翻转。若需要 `w2c`，统一求 `inverse(c2w)`，不混用参考工具中采用另一种行列布局的附带 `w2c` 返回值。

按用户选择，采用中心裁剪而非补边：

| 实验 | 数据加载 H×W | AnySplat patch | 实际输入、监督与评估 H×W | 裁剪 |
| --- | --- | --- | --- | --- |
| `omniscene_112x200` | 112×200 | 14 | **112×196** | 左右各 2 像素 |
| `omniscene_224x400` | 224×400 | 14 | **224×392** | 左右各 4 像素 |

DepthSplat 当前 shim 的倍数为 `shim_patch_size * downscale_factor = 4 * 4 = 16`，所以其同名低分辨率配置实际裁到 112×192，高分辨率 224×400 不变。复用的是裁剪策略，不能照搬数值 16；也不能把同名实验误写为完全相同的有效视野。

RGB、掩码、PCC 深度和所有目标视角必须用相同裁剪窗口。裁剪后的像素 K 要满足：`cx' = cx - left`、`cy' = cy - top`，再按新宽高归一化；焦距同步转换。DepthSplat 当前 shim 只显式调整归一化焦距，对非中心主点不够完整；本项目复用中心裁剪方法时完整更新主点，避免直接照搬这一遗漏。不会检查主点是否“合理”而筛数据。

不将裁剪后的输出拉伸回原宽度伪装成原视野评估。结果文件同时记录 `loaded_resolution` 和 `effective_resolution`；与 SVF-GS、DepthSplat 对比时说明有效视野差异。若以后需要严格相同像素域，应另行统一三项目评估区域，本轮不修改两个参考项目。

AnySplat 的现有 data shim 还会把 context RGB 从 `[0,1]` 变为 `[-1,1]`，wrapper 再转换回 `[0,1]` 才送入模型。适配应只执行一次这组约定，不能重复归一化，也不能把 `[-1,1]` 直接送入 VGGT。

### 3.4 动态物体掩码的确切含义

同一输入时刻的运动物体属于当前场景，应被重建。前后 12 视角中的物体位移，使其 RGB 对当前时刻的静态高斯构成不一致监督；SVF-GS 使用动态掩码正是为了减轻这种干扰。

本实验按用户确认保留 `use_dynamic_mask: true` 及参考加载约定：输入 6 路构造全 1；目标前 12 路在需要完整目标包时读取已有真实掩码，目标最后 6 路仍为全 1。训练仅加载输入 6 路时，无须为从不参加损失的新视角额外读取 RGB 或掩码文件。

RGB 损失的使用方式与 SVF-GS 一致：MSE 对 `GT * M` 与 `pred * M` 求全图均值，LPIPS 对相同的乘掩码图像计算。因为此实验参与监督的 M 恒为 1，损失等价于原 AnySplat 的输入视角全图 RGB 监督。这是正确的场景定义，不是掩码功能缺失；不改为遮掉输入图像中的运动物体。

不复制 DepthSplat 当前对 mask 转 bool 后取反的训练路径，也不将动态掩码代替 VGGT 置信度。正式评估遵循 SVF-GS nuScenes 的全图口径，不因加载了训练掩码而改变评估像素域。

### 3.5 能复用哪些加载代码

| 部分 | 处理 |
| --- | --- |
| bin 清单、相机顺序、前 12 后 6 拼接 | 复用 DepthSplat 的实现逻辑 |
| 小图路径、内参缩放、相机 OpenCV 约定 | 复用，并保持标准 `c2w` 表示 |
| DA2 文件读取与变换 | 复用，见第 7 节 |
| 中心裁剪 | 复用策略，倍数改为 14，补齐主点变换 |
| 掩码 | 以 SVF-GS 的含义为准，不复用 DepthSplat 的 bool 取反逻辑 |
| 训练加载 18 路目标 | 不沿用，AnySplat 训练只需要中心 6 路 |
| `LIDAR_TOP` 帧数读取及断言 | 不沿用，只按既定相机索引读取 |
| Dataset 配置注册 | 适配 AnySplat 的 wrapper union 与 dataset 映射 |
| DataLoader | 使用普通整数索引、固定 batch size 1，不沿用 AnySplat 的动态混合采样器 |

当前 AnySplat 的 `MixedBatchSampler` 明确忽略配置的训练 batch size，动态决定视角数、图像数和分辨率；`CustomConcatDataset`、测试 wrapper 还会把索引改成 tuple。因此只新增一个 dataset 文件并填 `batch_size: 1` 不够。OmniScene 必须走独立的固定加载路径，训练 shuffle，验证/测试固定顺序，禁用动态分辨率和动态视角数。

## 4. 主程序调用与信息隔离

### 4.1 数据包与调用边界

数据包保留 AnySplat 常用的 `context`、`target`、`scene` 结构，以减少日志和损失接口改动：

```text
context.image       [1, 6, 3, H, W]   唯一传给重建函数的外部张量
context.masks       [1, 6, H, W]      全 1，只用于 RGB 损失
context.intrinsics  [1, 6, 3, 3]      评估元数据；训练可以不加载
context.extrinsics  [1, 6, 4, 4]      评估元数据；训练可以不加载
target.image        [1, 18, 3, H, W]  仅测试目标真值
target.intrinsics / extrinsics       仅测试渲染相机
target.rel_depth    [1, 18, H, W]     仅 PCC
scene / bin_token                    日志与结果标识
```

训练包可以不含 target；后续任何 loss 都不得通过 target 获取监督。几何损失所需教师掩码在网络内部保存，不复用 `context.masks` 字段。

训练调用保持 `model(context_image, global_step)` 的图像接口；其内部 encoder 预测高斯及相机，再由 decoder 渲染输入视角。评估增加统一重建函数，仅调用 encoder 的完整高斯构建路径，返回 `gaussians` 与 `pred_context_pose`，然后在网络外对齐相机、渲染和评分。

DepthSplat 的 `encoder(context_dict, ...)` 会使用已知内外参构造几何关系；AnySplat 的 encoder 接收 `image_tensor` 并自行预测几何。因此不能复用 DepthSplat 的 encoder 调用语句，更不能为了接口方便把包含 K/T/depth 的字典变成 AnySplat 新输入。

### 4.2 6 路输入相机的全局相似对齐

AnySplat 的高斯位于其预测坐标系，数据集相机位于数据集坐标系，不能直接混用。每个 bin 在重建结束后，用 6 对输入相机估计唯一的正尺度全局变换：

```text
x_dataset = s * R * x_pred + t
```

拟使用相机姿态辅助的闭式相似对齐：由六对输入相机朝向的旋转差做 SO(3) 投影得到全局 R，再由六对相机中心的中心化最小二乘求正尺度 s，最后由中心均值求 t。采用固定六对相机，不用 RANSAC 剔除视角，不读取目标 RGB 拟合。数值退化属于模型输出处理，不触发数据审查：使用固定的单位尺度等数值回退并记录标记，保留该 bin 的评估，不静默删除困难样本。

保持高斯及球谐系数不变，将 18 路已知目标相机变换到预测坐标系。若目标相机在数据集中的旋转与中心为 `R_cam`、`c_cam`，则：

```text
R_cam_pred = R.T @ R_cam
c_cam_pred = R.T @ (c_cam - t) / s
```

目标 K 采用按 resize/crop 更新后的已知 K。该方法只改变评估坐标表达，不把输入相机作为网络条件，不用 12 个新视角估计变换，也不修正各输入相机之间的相对预测误差。所有 18 视角统一使用这一渲染规则。

原 `test.align_pose: true` 入口包含逐视角测试优化，不用于本实验；设为 false，并用新增 `camera_alignment: input_sim3` 明确区分。禁止用新视角 RGB 优化相机或高斯后再汇报前馈结果。

### 4.3 统一评估入口

增加一个共享评估流程，训练中 mini、训练结束 mini、独立 mini/total、作者权重评估均调用它：

```text
固定 6 路 RGB → 裁剪/归一化 → AnySplat 完整重建
           → 输入相机全局对齐 → 18 路 RGB＋深度渲染
           → all_18 / novel_12 / input_6 指标 → 逐 bin 保存与汇总
```

可以参考 DepthSplat 的独立测试与训练中评估组织，但不能直接复制其 wrapper：其 encoder 入参、监督目标和本项目不同，现有 AnySplat 测试代码也含旧返回值索引和其他数据集字段假设，需要针对 OmniScene 接通当前 `EncoderOutput`。

## 5. 训练、验证、mini 测试及自动续训

### 5.1 节奏

以完成的 optimizer step 计数，训练上限为 **100001**，不是 100001 个 epoch，也不是含验证在内的循环次数。

| 事件 | 时刻与内容 |
| --- | --- |
| 启动 | 解析配置、恢复完整状态或 VGGT 初始化、记录参数量、发送飞书启动日志 |
| 训练 | 每步 1 个 bin，固定中心 6 路图像，使用第 2.3 节损失 |
| 保存 | 沿用选定 AnySplat 配置每 200 步保存；保留最近 5 份及 `last.ckpt` |
| 验证 | 每 1000 步一次，固定 val 10 个 bin；不参与梯度 |
| 训练中 mini | 每 10 次正式验证一次，即第 10000、20000、…、100000 步 |
| 结束 | 第 **100001** 步保存最终完整 checkpoint，**额外执行一次最终 mini**，保存结果并推送飞书 |
| 独立 total | 用最终模型另行执行完整验证清单评估，作为正式对比结果 |

AnySplat 作者 main 没有周期 mini 功能。本分支已通过共享评估器补齐每 10 次验证调用及最终 mini，参考 DepthSplat 的训练中评估组织。

周期 mini 与最终 mini 的触发步数、完成状态必须持久化。100000 步的 mini 不能替代 100001 步的最终 mini。不要在正在运行的 Lightning `fit` 内重入同一个 Trainer 的 `test`；使用独立评估循环，在主进程完成后继续训练。

评估期间使用 `no_grad`/`inference_mode`，临时关闭教师执行，结束后恢复模型模式与原 `distill` 状态。当前 `AnySplat.inference()` 会直接把 `encoder.distill=False` 留在对象上，因此不能直接调用后继续训练而不恢复。mini 也不能永久改变 `requires_grad`。

### 5.2 自动续训

恢复优先级为：显式指定的完整训练 checkpoint → 本实验工作目录的最新完整 checkpoint → VGGT 初始化。

1. 只搜索当前实验的 `work_dirs/<experiment>/checkpoints`，优先 `last.ckpt`，必要时按实际 step 选择已完成写入的备份，不靠文件名字符串排序。
2. 通过 `trainer.fit(..., ckpt_path=...)` 恢复模型、optimizer、scheduler、global step 和训练循环状态；还需保存/恢复随机数、采样顺序与已完成 mini 标记，不能仅加载 `state_dict` 冒充断点续训。
3. 保存采用临时文件再原子替换，避免中断写坏 last；这属于 checkpoint 完整性处理，不是数据集审查。
4. 两个分辨率分别恢复，已有 work dir 不被覆盖成一次新的随机初始化实验。
5. 训练已到 100001 而最终 mini 尚未完成时，仅补跑最终 mini，不继续更新参数；两项都完成时记录完成状态，不自动重训。

保存最终 `final.ckpt`、对应配置与 `run_state.json`，并在 mini 汇总成功后记录 `final_mini_complete`。如果评估中断，保留模型与已有输出，重新评估不会触发重新训练。

### 5.3 离线日志与飞书

W&B 设为 `offline`，日志写入当前 work dir。运行所需 VGGT、LPIPS 及作者模型权重提前下载或从本地缓存加载，实际训练和评估期间避免隐式在线取权重；这是模型准备，不是数据集离线信息生成。

按 SVF-GS 加载方式，从可配置的 `library_root`（本机默认 `~/Libraries`）导入：

```python
from auto_monitor.send_feishu import send_feishu
send_feishu(subject, content)
```

仅主进程在以下事件推送：

- 训练启动/续训启动：实验名、两个尺寸、工作目录、初始化或恢复路径、当前步数、目标步数、参数量。
- 每次 mini 完成，包括最终 mini：step、样本数、`all_18` 与 `novel_12` 的四项指标、重建耗时、耗时口径、checkpoint、剩余时间估计及结果目录。

不复制密钥或 webhook 到仓库。网络发送设置有限等待，失败写入本地日志但不阻塞训练；正式运行默认启用，本轮调试显式关闭推送，没有发送实际通知。PSNR 显示 3 位小数，SSIM/LPIPS/PCC 显示 4 位小数，原始结果保留完整精度。

## 6. 作者预训练权重直接评估

当前仓库公开用法为 `AnySplat.from_pretrained("lhjiang/anysplat")`。本方案提供本地作者模型目录加载模式，读取其配置与权重，评估时 `distill=false`，不建立用于训练的教师副本，不启动 optimizer，不进行 OmniScene 微调。

两套尺寸都支持以下两类模型：

| 模型来源 | OmniScene 训练 | mini/total |
| --- | --- | --- |
| 本实验 `final.ckpt` | VGGT 初始化后训练 100001 步 | 均支持 |
| 作者 `lhjiang/anysplat` | 无，直接评估 | 均支持 |

作者 safetensors 模型目录与 Lightning 完整 checkpoint 分开加载。前者不能作为 `trainer.fit(ckpt_path=...)` 的续训文件，后者不能作为 Hugging Face 模型目录。实现时显式映射训练 checkpoint 的学生模块，检查模型键的兼容性，不用无条件 `strict=False` 隐藏加载遗漏。构造器原有 VGGT 初始化依赖也应通过本地缓存或专门加载路径处理，避免离线评估意外联网。

在已安装依赖的 AnySplat 环境中，从项目根目录执行以下命令：

```bash
# 两套训练；工作目录内有完整断点时自动恢复。
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=omniscene_112x200
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=omniscene_224x400

# 自训模型的完整测试；mini 只需把 test.split 改为 mini。
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=omniscene_112x200 mode=test test.split=total test.weights_source=training checkpointing.load=work_dirs/omniscene_112x200/checkpoints/final.ckpt hydra.run.dir=work_dirs/omniscene_112x200/eval/total
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=omniscene_224x400 mode=test test.split=total test.weights_source=training checkpointing.load=work_dirs/omniscene_224x400/checkpoints/final.ckpt hydra.run.dir=work_dirs/omniscene_224x400/eval/total

# 作者权重直接评估：独立目录，不自动加载同名训练断点。
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=omniscene_112x200 mode=test test.split=total test.weights_source=author test.pretrained_path=pretrained hydra.run.dir=work_dirs/author_112x200/eval/total
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=omniscene_224x400 mode=test test.split=total test.weights_source=author test.pretrained_path=pretrained hydra.run.dir=work_dirs/author_224x400/eval/total
```

`mode=test` 时不执行训练自动续训搜索；按 `weights_source` 和显式权重来源加载，避免“作者模型评估”实际载入本地自训模型。

## 7. PCC、图像指标与统计

### 7.1 相对深度加载

复用 DepthSplat `src/dataset/utils_omniscene.py` 的 DA2 处理，与 SVF-GS 一致：

1. 用 RGB 路径找到 `samples_dpt_small` / `sweeps_dpt_small` 下的 `.npy`。
2. 将 disparity 按加载 H×W 双线性缩放。
3. 按参考实现将最远/最近比限制为 50：`ratio=min(disp.max()/(disp.min()+0.001),50)`，`floor=disp.max()/ratio`。
4. `depth=1/max(disp,floor)`，每张图 min-max 归一化为 `[0,1]`。
5. 与 RGB 执行相同中心裁剪；拼接为前 12 路＋输入 6 路的 `target.rel_depth`。

顺序保持“resize、disparity 转换与归一化、裁剪”，不能为了方便改成另一顺序。只在 mini/total 评估加载 DA2，不把它接入教师、训练损失、尺度标定或高斯构建。

### 7.2 渲染深度

AnySplat 当前 decoder 已通过 gsplat 的 `RGB+D` 返回 `DecoderOutput.depth`，因此无需额外跑一个目标图像深度网络。一次目标渲染同时得到 RGB 与深度，整理为 `[1,18,H,W]` 与参考深度对应。

保持原 `D` 深度模式，不暗中改成 `ED`、alpha 归一化深度或 inverse depth。PCC 使用这一深度与 DA2 相对深度计算；不引入 Metric3D 尺度校正，也不用 DA2 拟合深度后再评分。全局正尺度对 Pearson 相关性不构成尺度误差，但深度模式仍必须记录。

### 7.3 计算位置与分组汇总

在共享评估器得到一整个 bin 的 18 路渲染后计算指标，三种调用路径（训练中、结束时、独立测试）使用同一段代码。

| 指标 | 单 bin、单视角组计算方式 |
| --- | --- |
| PSNR | 逐视角计算 PSNR，再在组内取均值；不是先合并全部像素 MSE 再取对数 |
| SSIM | 沿用现有函数：`win_size=11`、Gaussian weights、`data_range=1`，逐视角后组内均值 |
| LPIPS | VGG，`normalize=True`，逐视角后组内均值 |
| PCC | 将该组的全部视角深度展平为一个向量，计算一次 Pearson 相关系数 |

因此 PCC 是：`corr(gt_depth[group].reshape(-1), pred_depth[group].reshape(-1))`。`all_18` 和 `novel_12` 分别计算，不能把每视角 PCC 平均冒充参考口径，也不能由三组数值按视角数加权推导另一个 PCC。

可以移植 DepthSplat 的 `compute_pcc`，确保每次调用只计算当前 bin 的当前组，避免有状态指标对象将前一次组/样本累计进去。分组与结果组织参考 SVF-GS `tools/ablation_metrics.py`，不复制其中与本任务无关的其他数据集 mask 审核逻辑。

数据集最终指标为**逐 bin 指标的算术平均**，每个 bin 等权。不得把整个测试集深度拼接后只计算一次全局 PCC。训练动态掩码不作用于正式 PSNR/SSIM/LPIPS/PCC，不使用天空、车辆自遮挡或其他新增掩码。

数学上未定义的相关系数不能被伪造成 0，也不能触发数据审核中断。保留该条记录、标注未定义原因与数量；若出现此情况，汇总必须显式展示，不能静默丢弃后仍声称全样本 PCC。

### 7.4 输出文件

每次评估在独立目录保存：

```text
metrics_per_bin.csv          # bin_token、view_group、四项指标
evaluation_summary.json     # 分组均值、实际 bin 数、split、完成状态
reconstruction_time.json    # 完整重建计时与边界说明
model_parameters.json       # 学生/教师、可训练/冻结/总参数
resolved_config.yaml        # 本次配置
run_metadata.json           # 代码版本、权重来源、尺寸、相机对齐、计时硬件
```

与 DepthSplat 旧 `scores_pcc_all.json` / `scores_all_avg.json` 的数值公式保持一致，但扩展分组字段，不能只输出一个没有 `all_18`/`novel_12` 标签的 PCC。总表还需区分官方模型、自训模型及 mini/total；训练中 mini 不能替代正式 total 结果。

## 8. 参数量与完整重建耗时

### 8.1 参数量

从实际实例化、加载后的模型统计，按参数对象去重，不抄论文约数，也不将两个分辨率称为两个参数规模。报告：

| 范围 | 内容 |
| --- | --- |
| 重建网络 | 学生 aggregator、camera/depth head、高斯预测头及高斯构建中实际持有参数的模块 |
| 重建网络可训练参数 | 按本实验冻结策略 `requires_grad=True` 的唯一参数数目 |
| 重建网络冻结参数 | 重建网络中其余参数 |
| 重建网络总参数 | 上述两者之和 |
| 训练教师 | 单独报告冻结 VGGT 副本参数，不混入部署重建网络 |
| 训练模型合计 | 学生＋教师的可训练、冻结、总参数，解释训练资源占用 |

LPIPS 等损失/指标网络单独列为辅助网络，不算进重建网络。测试时 `eval()` 不代表模型参数定义变成零个可训练参数；记录训练配置下的冻结划分，另注明此次评估不更新任何参数。官方模型同时记录其实际加载结构及本实验采用的计数定义。

### 8.2 计时范围

主要指标为 `reconstruction_ms_per_bin`：batch size 1、6 路图像，从 GPU 上图像交给完整重建入口开始，到可直接渲染的高斯全部构建完毕为止。包含入口内尺寸/归一化处理、VGGT 学生前向、相机与深度预测、反投影、高斯属性生成、体素融合及最终激活/协方差构造；不能只计 backbone 或高斯头。

计时前后执行 CUDA 同步，使用 wall-clock 计时，因此包含入口内的 CPU 调度与 CUDA kernel 工作。前 5 个 bin 仅作计时预热，它们仍参加全部质量指标。报告平均值、中位数、P95、计时样本数、GPU 型号与是否有其他任务共享 GPU。

主要重建计时不含文件读取、DataLoader 等待、CPU→GPU 传输、冻结教师、相机对齐、18 路渲染、PCC/图像指标和保存文件；逐项记录边界。另保存 `reconstruction_with_transfer_ms_per_bin`，从 CPU RGB 已准备好、开始送入设备前计时至高斯完成，明确给出包含 H2D 的完整入口耗时。实现中避免 Lightning 自动迁移把 H2D 悄悄移到第二个计时窗口之外。

相机对齐、渲染和评分是重建完成后的评估步骤。若另报它们的耗时，单独命名，不把重建耗时称为“18 视角端到端渲染耗时”。mini 中的计时用于进度观察，正式耗时由独立评估在记录清楚的硬件环境下产生。

## 9. 实现位置与检查

| 文件/模块 | 计划改动 |
| --- | --- |
| `config/dataset/omniscene.yaml`、两套 experiment | 数据、实验与节奏配置 |
| `src/dataset/dataset_omniscene.py`、配套读取工具 | 固定 6/18 视角，划分，掩码，DA2 深度 |
| `src/dataset/__init__.py`、`src/config.py` | Dataset wrapper、typed config 与新增运行设置 |
| `src/dataset/omniscene_data_module.py` | 固定 batch size 1 的 DataLoader 与可恢复采样器 |
| dataset shim | 按 patch 14 中心裁剪，同步 K、mask、rel_depth |
| `src/main.py`、`src/omniscene.py` | 固定工作目录与三条权重加载路径 |
| `src/model/omniscene_wrapper.py`、`src/misc/omniscene_callbacks.py` | context-only 损失、验证、mini 触发与状态恢复 |
| `src/loss/loss_distill.py` 及必要 RGB loss | 跳过零权重法线计算；独立 RGB mask，不覆盖教师 confidence mask |
| 新增共享 OmniScene evaluator | 完整重建、input Sim(3)、18 路渲染、分组指标、结果汇总 |
| `src/evaluation/metrics.py` | 移植 PCC；不改变已兼容的 RGB 指标公式 |
| checkpoint/通知/统计工具 | 自动恢复、send_feishu、参数量、计时 |

实现检查重点是程序和信息流，而非数据集质量：

- 配置解析后确实是单数据源、batch 1、100001 步、每 1000 步验证、W&B offline。
- 两套尺寸得到上述有效裁剪尺寸，RGB/深度/mask 像素与相机主点同步。
- 用可控小样例检查：只改变目标 RGB 或外部 K/T/Metric3D，不能改变重建函数输出；新视角真值只影响评估分数。
- 原输入视角损失及内部几何监督保留，输入动态掩码全 1，几何置信度掩码没有被覆盖。
- 同一缓存渲染结果经移植 PCC 与参考 PCC 得到一致结果，并分别产生 `all_18`、`novel_12`。
- 中断恢复继续原 step/优化器/调度器；mini 不永久关闭 distill 或冻结学生；100001 步最终 mini 被执行且不重复训练。
- 官方权重入口不误加载 work dir 自训模型；完整重建计时覆盖体素融合等后处理。

实际检查包含 CPU 回归测试和 GPU 短程调试，详见第 11 节。没有修改 SVF-GS 或 DepthSplat。

## 10. 主要代码依据

以下是本方案实际核查的入口，方便后续实现与审阅追溯：

- AnySplat：`config/experiment/multi-dataset.yaml`、`dl3dv.yaml`；`config/model/encoder/anysplat.yaml`；`src/main.py`；`src/dataset/data_module.py` 与 `__init__.py`；`src/model/model/anysplat.py`；`src/model/encoder/anysplat.py`；`src/model/model_wrapper.py`；`src/model/decoder/decoder_splatting_cuda.py`；`src/loss/loss_depth_consis.py`、`loss_distill.py`；`src/evaluation/metrics.py`。
- SVF-GS：`data/omniscene_dataset.py`；`data/transforms/loading.py`；`model/omni_gs.py::compute_loss`；`trainer.py`；`tools/metrics.py`；`tools/ablation_metrics.py`。
- DepthSplat：`config/experiment/omniscene_112x200.yaml`、`omniscene_224x400.yaml`；`src/dataset/dataset_omniscene.py`、`utils_omniscene.py`、`shims/patch_shim.py`；`src/model/model_wrapper.py`；`src/evaluation/metrics.py`；`src/misc/resume_ckpt.py`。

DepthSplat 自带的早期 OmniScene 文档与当前 PCC、测试划分实现有出入，因此本方案没有照搬其中“尚未实现 PCC”等旧描述。


## 11. 实现与调试说明

### 11.1 已接通的路径

- `src/main.py` 检测 `dataset.omniscene` 后调用 `src/omniscene.py`，作者原数据集仍走原入口。两套分辨率配置继承 `config/experiment/omniscene_common.yaml`。
- 固定数据加载器按已完成的 optimizer step 恢复采样位置，不用 DataLoader 预取进度当作已经训练的样本数。内部按固定种子逐轮打乱，外部按总步数训练。
- checkpoint 保存完整模型、教师、优化器、调度器、Lightning 循环、随机数及采样位置。`last.ckpt` / `final.ckpt` 使用原子硬链接更新，避免同一大型断点重复占用磁盘。
- 验证输出输入 RGB 与渲染 RGB 的对比图，保存在实验目录 `val/step-XXXXXXXX/`。
- mini 和独立评估共用 `src/evaluation/omniscene.py`，写入逐 bin 指标、分组汇总、相机对齐记录、参数量和两种重建计时。
- 训练结束后先保存完整最终断点，再将学生网络放回 GPU 执行最终 mini；教师留在 CPU。已完成训练而最终 mini 未完成时，可直接恢复并补跑。
- 本机作者模型位于 `pretrained/config.json` 与 `pretrained/model.safetensors`，实际默认目录为 `pretrained`。

VGGT 初始化默认从本地 Hugging Face 缓存读取，配置 `model.encoder.local_files_only=true`。如换机器，应先准备 VGGT 缓存、LPIPS/VGG 权重和作者模型目录，或把 `model.encoder.vggt_pretrained_path` 指向本地 VGGT 目录；训练期间不会为数据集生成额外离线信息。

### 11.2 调试边界与已有检查

所有本轮调试日志、checkpoint、W&B 离线目录、渲染图像和指标均位于 `/tmp/anysplat-omniscene-debug-Qtfm6h/`。项目 `output*`、`outputs`、`work_dirs` 没有写入调试结果。调试期间飞书关闭。

CPU 回归测试位于 `tests/test_omniscene.py`，覆盖配置、仅 6 路 RGB 即可训练的数据读取、18 路顺序、掩码约定、裁剪后的相机射线、Sim(3)、PCC 统计、零权重法线不依赖 K，以及 Lightning 完整续训。续训测试比较连续运行与中断恢复后的参数、学习率状态和样本顺序，包括 Python/NumPy/PyTorch 随机数的影响。

GPU 调试使用真实 OmniScene 样本与真实模型，已覆盖两套分辨率的作者权重推理、训练反向传播、验证、周期 mini、最终 mini，以及自训 checkpoint 独立评估。112×200 还验证了自动恢复第 2 步后继续到第 3 步，以及多进程数据加载。调试只运行少量训练步骤，评估使用 `test.limit_batches=1`；这类结果明确标记 `complete: false`，不能视为 mini 或 total 正式结果。正式配置默认 `test.limit_batches: null`，mini 仍为既定列表，total 为完整验证列表。

测试命令（建议设置 `PYTHONNOUSERSITE=1`，避免用户目录的其他 PyTorch 版本污染环境）：

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_omniscene.py' -v
```

如需自己做临时调试，应同时指定 `hydra.run.dir=/tmp/<debug-name>`、`notifications.feishu_enabled=false`，并显式设置较小的训练步数或 `test.limit_batches`。正式实验使用第 6 节的命令，不携带这些调试覆盖项。


## 12. 首轮实验结果与分析（2026-10-03）

### 12.1 实验身份、停止状态与记录来源

本节记录的是一次未完成的训练尝试及其训练中 mini 评估，不能作为训练完成后的 total 对比结果。实验沿用前文确认的 `multi-dataset` 配置、VGGT 初始化、仅输入 6 路 RGB 的重建网络，以及只监督输入视角的训练方式。

| 项目 | 本轮记录 |
| --- | --- |
| 实验名 | `omniscene_112x200` |
| 加载 / 实际处理分辨率 | 112×200 / 112×196 |
| 初始化 | `facebook/VGGT-1B`；未以作者 AnySplat 成品模型初始化训练 |
| 已交付适配代码 | `366c718ec1876716d7556c0b6adca11fcdecfd20`，`comp_svfgs`，中文提交“新增 OmniScene 对比实验训练与评估支持” |
| 启动时 Git HEAD | `9f1c4aeccc3b96947478be584fdb9aebe94512d5`；训练启动时适配代码尚未提交，因此不能把它理解为运行了未经适配的作者版本 |
| 启动时实现内容 SHA-256 | `a564361749543a011fe5c20db7826d59ad0d770f8f140c45ff0f11743c74640b`，由 `src/**/*.py` 与 `config/**/*.yaml` 的路径及内容计算 |
| 计划步数 | 100001 |
| 最后观察到的训练日志 | 2026-10-03 16:01:12，第 62880 步，日志每 10 步记录一次；不将其当作精确停止步数 |
| 停止状态 | 已观察到用户 Ctrl+C、`KeyboardInterrupt`，tmux 已回到 zsh；未由助手停止训练 |
| 最后保存的检查点 | `step-00062800.ckpt`，与 `last.ckpt` 为同一文件的硬链接；迁移后已只读加载，内部 `global_step=62800`，包含模型、优化器、学习率调度器、Lightning 循环及随机数状态；未实际重新启动训练验证恢复 |
| 已完成常规 val | 62 次，每次固定 10 个样本；最后一次为 62000 步 |
| 已完成训练中 mini | 10000、20000、30000、40000、50000、60000 步各一次，每次 2048/2048 个 bin |
| 最终 mini / total | 均未执行 |
| 224×400 正式训练 | 本轮未启动；此前只有第 11 节所述短程调试 |

用户已将正式产物迁移到数据盘，项目内 `work_dirs` 现为指向 `/home/C_UserData/dongzhipeng/Datasets/Projects/AnySplat/work_dirs` 的软链接。迁移后已确认 `work_dirs/omniscene_112x200/`、六次 mini 汇总及检查点可读。`run_state.json` 的 `last_mini_step=60000`、`training_complete=false`、`final_mini_complete=false`；其中 `step=60000` 是该运行状态文件记录的评估进度，不是最新检查点步数。

迁移期间，16:02:26 完成一次只读结果快照，保存在 `/tmp/anysplat-results-record-20261003.json`，含 val 历史、六次 mini 汇总、相机对齐回退计数及计时。本节将关键结果转录到版本管理中的文档；正式原始记录仍在上述数据盘目录，`/tmp` 快照和诊断文件只是临时证据，不保证永久存在。助手没有删除、移动或覆盖训练产物。

### 12.2 首轮训练监督及 val 的解释

首轮有效损失为 `RGB MSE × 1.0 + LPIPS × 0.05 + 深度一致性 MSE × 0.1 + 相机参数蒸馏 × 10.0`。深度一致性比较学生深度头与高斯渲染深度；相机蒸馏目标为冻结 VGGT 的预测。直接深度蒸馏和法线蒸馏权重均为 0。

训练与常规 val 使用**网络预测的相机内外参**渲染输入 6 路；mini 使用**经输入相机全局 Sim(3) 对齐后的数据集相机外参及真实内参**。因此两者虽然都能包含输入 6 路 RGB 真值，但相机条件不同，PSNR 不能当作同一种几何评估直接比较。

| val 步数 | 输入视角 PSNR（dB） | 总损失 |
| --- | ---: | ---: |
| 1000 | 19.295 | 0.054240 |
| 10000 | 22.044 | 0.047911 |
| 20000 | 21.243 | 0.053603 |
| 30000 | 21.450 | 0.052525 |
| 38000 | 21.404 | 0.047868 |
| 40000 | 21.002 | 0.052048 |
| 50000 | 20.942 | 0.067192 |
| 60000 | 21.660 | 0.070880 |
| 62000 | 21.796 | 0.065620 |

W&B 的该条验证记录 `trainer/global_step` 为上表步数减 1；验证输出目录及验证触发次数采用已完成更新数，如 `val/step-00062000/`。总损失与分项日志均已包含对应权重。

首轮 val 每个样本保存一张 6 路输入 RGB / 渲染 RGB 对比图，每轮共 10 张。**未保存深度可视化或原始深度数组**。观察到后期渲染能恢复部分颜色、道路与建筑轮廓，但仍有明显模糊、拉丝及结构错误；不能仅从 RGB 图确认深度是否为常数。

### 12.3 六次完整 mini 评估

下表来自各轮 `eval/mini-step-XXXXXXXX/evaluation_summary.json`。所有轮次均为 `complete: true`、2048/2048 个 bin，且未设置调试 batch 截断。`complete` 仅表示样本处理完成，不表示重建质量合格。PSNR 保留 3 位小数，SSIM / LPIPS 保留 4 位小数。

| 步数 | 视角组 | PSNR（dB） | SSIM | LPIPS | PCC 汇总 | PCC 未定义 bin 数 / 2048 |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| 10000 | `all_18` | 3.983 | 0.2688 | 0.7285 | 未定义（JSON null） | 1324 |
| 10000 | `novel_12` | 3.986 | 0.2683 | 0.7289 | 未定义（JSON null） | 977 |
| 10000 | `input_6` | 3.978 | 0.2699 | 0.7278 | 未定义（JSON null） | 1912 |
| 20000 | `all_18` | 3.987 | 0.2687 | 0.7285 | 未定义（JSON null） | 1322 |
| 20000 | `novel_12` | 3.987 | 0.2682 | 0.7288 | 未定义（JSON null） | 949 |
| 20000 | `input_6` | 3.986 | 0.2696 | 0.7279 | 未定义（JSON null） | 1805 |
| 30000 | `all_18` | 3.984 | 0.2687 | 0.7284 | 未定义（JSON null） | 1292 |
| 30000 | `novel_12` | 3.987 | 0.2681 | 0.7288 | 未定义（JSON null） | 925 |
| 30000 | `input_6` | 3.980 | 0.2697 | 0.7277 | 未定义（JSON null） | 1867 |
| 40000 | `all_18` | 4.006 | 0.2670 | 0.7275 | 未定义（JSON null） | 818 |
| 40000 | `novel_12` | 4.007 | 0.2665 | 0.7278 | 未定义（JSON null） | 494 |
| 40000 | `input_6` | 4.004 | 0.2679 | 0.7271 | 未定义（JSON null） | 1455 |
| 50000 | `all_18` | 4.016 | 0.2664 | 0.7268 | 未定义（JSON null） | 934 |
| 50000 | `novel_12` | 4.015 | 0.2660 | 0.7272 | 未定义（JSON null） | 621 |
| 50000 | `input_6` | 4.016 | 0.2671 | 0.7260 | 未定义（JSON null） | 1530 |
| 60000 | `all_18` | 4.036 | 0.2644 | 0.7258 | 未定义（JSON null） | 766 |
| 60000 | `novel_12` | 4.038 | 0.2641 | 0.7262 | 未定义（JSON null） | 510 |
| 60000 | `input_6` | 4.032 | 0.2650 | 0.7249 | 未定义（JSON null） | 1345 |

PCC 汇总为 null 是因为该组包含未定义的逐 bin PCC，汇总代码没有静默丢弃这些 bin 后计算较好看的均值。这不等于每个 bin 的 PCC 都未定义，也不等于 PCC 为 0。PSNR、SSIM、LPIPS 在各轮汇总中均无未定义条目。

已知相机的全局对齐出现非正尺度时，原实现按既定规则回退到单位尺度、保留样本。各轮回退数量如下；这些是模型输出与对齐的诊断记录，不是对数据集的质量判定或筛选依据。

| mini 步数 | `nonpositive_scale_used_unit_scale` 数量 / 2048 |
| --- | ---: |
| 10000 | 752 |
| 20000 | 686 |
| 30000 | 718 |
| 40000 | 649 |
| 50000 | 600 |
| 60000 | 568 |

10k 到 60k，输入 6 路的真实相机 PSNR 仅从 3.978 dB 变化到 4.032 dB；全部 18 路与新视角 12 路也一直约为 4 dB。首轮没有呈现出真实标定下有效重建的改善。

### 12.4 同一检查点、同一高斯的相机对照诊断

诊断时间为 2026-10-03，检查点为 `step-00062400.ckpt`。固定选择 mini 列表索引 0、300、1000，各自使用同一组六路 RGB、同一次前馈产生的同一份高斯，只替换渲染器接收的相机内参和外参。三例均未触发尺度回退，说明空白渲染并非仅由回退样本造成。

| 渲染条件 | bin 0 PSNR | bin 300 PSNR | bin 1000 PSNR | 三例平均 PSNR（dB） |
| --- | ---: | ---: | ---: | ---: |
| 预测外参＋预测内参 | 18.646 | 19.282 | 18.834 | 18.921 |
| 预测外参＋数据集内参 | 10.831 | 12.395 | 10.943 | 11.390 |
| 全局对齐后的数据集外参＋预测内参 | 4.591 | 4.533 | 3.883 | 4.336 |
| 全局对齐后的数据集外参＋数据集内参（当前 mini 规则） | 4.651 | 4.533 | 3.883 | 4.356 |

诊断样本标识：

- mini 索引 0：`scenee7ef871f77f44331aefdebc24ec034b7_bin000`。
- mini 索引 300：`scene5af9c7f124d84e7e9ac729fafa40ea01_bin152`。
- mini 索引 1000：`scene0ac05652a4c44374998be876ba5cd6fd_bin121`。

进一步观测：

- 使用预测相机时，三例渲染 alpha 的均值约为 0.878、0.907、0.922，输入视角均有高斯覆盖。
- 换为当前 mini 的相机条件后，bin 300 和 bin 1000 的六路 alpha 全为 0，渲染 RGB 为白背景、渲染深度全为 0；bin 0 也仅有很少覆盖，alpha 均值约 0.043。这解释了抽查中的约 4 dB 和未定义 PCC，不能据三例推断所有未定义 PCC 都只有同一原因。
- 三例在全局对齐后，各输入相机的旋转残差仍约为 11°～159°。预测内参的纵向焦距约为数据集对应值的 2.1～4.7 倍。
- 已核对加载器的 OpenCV `c2w` 约定与 DepthSplat 一致，没有沿用其附带行向量布局的 `w2c`，也没有额外翻转 Y/Z。用已知正尺度 Sim(3) 构造六相机往返测试，最大绝对误差约 `8.88e-16`，未发现全局变换公式简单写反的问题。这些检查不是对整个实现无误的证明。
- 三例学生深度头输出均有空间变化，标准差非零；使用预测相机渲染时，深度也不是整幅常数。按用户约定，不能把深度均值缩小、PCC 下降或模糊图像直接称为“深度退化”。应区分深度头预测为常数与当前真实相机看不到高斯导致的零深度空渲染。

诊断是三个固定样本的机制对照，不是新的 mini / total 分数，不替代第 12.3 节结果。脚本、三例 JSON、RGB 对比图与原始深度 `.npy` 位于 `/tmp/anysplat-eval-diagnosis-20261003/`，总记录为 `summary.json`；没有修改网络权重、训练配置或正式评估口径。

### 12.5 当前分析结论与证据边界

1. **首轮在既定真实相机评估协议下未得到有效重建结果。** 输入 6 路和新视角 12 路均约为 4 dB，这已经不能仅用新视角较难解释。
2. **网络仍能在自身预测相机下恢复部分输入外观。** 约 21 dB 的常规 val 与三例约 18.921 dB 的诊断属于这一条件，不能将其作为真实标定下的 6 路输入或 18 路结果。
3. **当前主要证据指向预测相机及配套高斯几何与真实标定不一致。** 全局 Sim(3) 只调整整体旋转、平移和尺度，不能纠正各相机独立的相对误差、错误内参及局部场景结构。对照中仅替换外参即出现大幅下降，内参差异也有影响。
4. **不能据此宣称 AnySplat 普遍无法在 OmniScene 上工作。** 本轮仅验证了既定初始化、分辨率、损失和 RGB-only / context-only 适配；尚未得到作者权重完整 mini / total 的结论，也不能用三个样本排除全部潜在实现问题。
5. **不再以原样继续增加迭代作为解决路径。** 10k～60k 的 mini 结果没有实质改善；下一阶段优先考虑最小的相机几何兼容处理，并明确它与原作者模型输入条件的差异，而不是围绕本数据集持续调优。

此前 2026-10-02 对 200 / 400 步固定样本出现“深度均值 0.5684 → 0.0241、逐视角相关系数均值 0.5981 → 0.2403”的记录，只能说明尺度和空间质量变化。此前将其称为疑似深度退化的表述已撤回；用户所指深度退化是整幅图所有像素预测为同一个深度值，常见为最远值。

### 12.6 资源与运行附注

重建网络按本轮冻结策略统计：可训练参数 **886,363,720**，冻结参数 **304,372,736**，合计 **1,190,736,456**。冻结 VGGT 教师另计 **1,157,941,492**，训练模型合计 **2,348,677,948**；LPIPS 等辅助网络未计入。

60k mini 在 RTX 4090 上排除前 5 个预热 bin 后，2043 个 bin 的完整重建耗时为：均值 **70.551 ms/bin**、中位数 **70.543 ms/bin**、P95 **71.740 ms/bin**；含 H2D 的入口均值为 **70.650 ms/bin**。边界遵循第 8 节，排除相机对齐、渲染、评分及数据加载。该计时来自训练中 mini，未强制 GPU 独占，且本轮重建质量失败，不能当作质量达标后的最终部署性能。

初期每 1000 步约 14 分 47 秒～14 分 51 秒；后期约 16 分钟，包含常规 val 和检查点保存。每轮完整 mini 约 5 分 15 秒。训练停止后，此前预测的完成日期不再适用。

首轮 W&B 为离线模式。启动和 mini 完成时的飞书推送均出现失败日志，结果以本地记录为准。独立的每 5 分钟写日志监控脚本已按用户要求关闭，没有自动唤醒助手的监控任务。

## 13. 后续最小适配方案（2026-10-03 已确认，实现见第 14 节）

### 13.1 目的与已确认的实验边界

下一阶段只尝试消除已经观察到的相机条件不一致，使 AnySplat 能在既定真实相机协议下产出初步对比结果。一次适配后按固定配置评估，不开展学习率、损失权重、体素尺寸、网络结构或训练策略搜索。初步结果可以较差，不以追平 SVF-GS 或提高到某个 PSNR 为目标。

**采用已有 6 路输入相机的内外参参与深度反投影和高斯构建，并在训练、val 和测试中使用同一套相机几何。** 用户已于 2026-10-03 明确回复：“允许使用已知输入相机，采用最小适配方案”。这放宽了原先“完整重建网络仅输入 RGB”的约定。即使 RGB 特征骨干仍然只接收图像，也不能把改动后的整个重建系统称为 RGB-only。标记为 `AnySplat + 已知输入相机`，与第 12 节原始输入条件的尝试分开报告。

输入视角监督边界保持不变：仍只读取并监督同一时刻的 6 路输入 RGB，输入动态掩码仍为全 1；不以 12 路新视角 RGB 训练，不使用时序特征，不新增 Metric3D 监督，不获取任何额外离线信息。新视角相机只在评估渲染时使用；DA2 深度仍只用于 PCC。

原始 RGB-only 模式和作者预训练权重直接评估入口继续保留，作为独立条件。作者权重评估是零样本结果，不能代替本方案在 OmniScene 上训练后的结果；此前单 bin 接口调试不构成完整 mini 证据。当前选定的后续主线是已知输入相机适配，不增加其他调优分支。

### 13.2 唯一的模型适配：用已知输入相机构建并监督高斯

当前数据流为：`RGB → 预测深度、预测相机及高斯属性 → 按预测相机反投影 / 融合 → 用预测相机训练渲染`，而测试再切换到经 Sim(3) 对齐的真实相机。

建议的数据流为：`RGB → 原有深度头和高斯属性头 → 按已知输入相机反投影 / 融合 → 用相同的已知相机训练渲染`。测试时直接使用同一坐标系下的目标相机，不再根据预测相机估计 Sim(3)。不能只改测试相机或只改训练渲染相机、却继续用错误的预测相机构建高斯。

拟修改的位置和职责如下；这些是实现计划，不是已经存在的新接口：

| 位置 | 限定改动 |
| --- | --- |
| `src/dataset/dataset_omniscene.py` | 在新模式的训练 / val 中加载现有 6 路输入 K / c2w；仍不读取新视角 RGB 或深度标签。测试沿用现有 18 路数据和分组。 |
| `src/model/encoder/anysplat.py` | 给现有 `depth → pts_all` 反投影增加已知相机路径；维持原深度头、高斯参数头、体素融合和 Gaussian adapter。预测相机与实际构建相机分别记录，不能把已知值写成预测值。 |
| `src/model/model/anysplat.py`、`src/model/omniscene_wrapper.py` | 传递 6 路已知相机，并用构建高斯时相同的相机渲染输入监督图像。 |
| `src/evaluation/omniscene.py` 与重建计时入口 | 新模式直接在统一坐标系渲染目标 18 路；把输入 K / c2w 的传输与相机规范化纳入相应重建计时边界。保留原 RGB-only 模式的 Sim(3) 路径。 |
| 实验配置、检查点协议记录 | 用独立模式和实验名区分输入条件，防止旧实验自动续训到新协议；原作者权重直接评估入口继续保持原始 RGB-only 条件。 |

几何约定必须一次固定：沿用 OpenCV 相机坐标；内参经过既有 resize 和中心裁剪，再由归一化内参还原到反投影所需的像素内参；反投影函数使用 w2c，渲染接口使用 c2w。使用第一个输入相机作为参考坐标系，并按六个输入相机中心到其质心的平均距离归一化平移，使该平均距离为 1；同一个变换应用于所有评估相机。这个尺度仅由 6 路输入标定确定，作为固定坐标约定，不利用新视角图像或深度拟合尺度，也不搜索尺度系数。网络深度及渲染深度在该约定下比较，不宣称为米制深度。

训练保留 VGGT 初始化、原冻结策略和所有既定损失权重：MSE 1.0、LPIPS 0.05、深度一致性 0.1、相机蒸馏 10.0、直接深度蒸馏 0、法线蒸馏 0。相机头及其蒸馏保留，但新模式下它的预测不再决定反投影和渲染相机；不新增相机真值损失。深度一致性仍比较学生深度与同一相机条件下的渲染深度。相机蒸馏仍可能通过共享骨干影响训练，不能将这一路影响描述为已经消失。

该修改直接处理观测到的相机误差，**不保证深度或新视角质量随之达标**；六路输入监督和场景覆盖的原有限制仍然存在。若仍失败，记录结果，不继续叠加深度监督或改造模型。

### 13.3 固定执行顺序与停止范围

1. **保留首轮产物。** 保留当前数据盘上的检查点和全部评估，不删除、不覆盖。新协议使用新实验名，例如 `omniscene_knowncam_112x200` 和 `omniscene_knowncam_224x400`，均从原定 VGGT 初始化重新训练；不继承首轮 62800 步的优化器、调度器或高斯头状态。新实验内部仍可自动断点续训。
2. **只做必要的几何和运行检查。** 检查已知射线反投影再投影、相机变换往返、训练 / val / 测试相机一致，以及 6 路监督的数据边界。用前述固定样本检查 RGB、深度和 alpha；所有诊断图、日志、缓存和临时检查点放 `/tmp`。这些检查针对实现和模型输出，不筛选或审查 OmniScene 数据。
3. **一次有上限的短程检查。** 112×200 最多训练 1000 步并运行一次固定 mini，以确认运行流程和真实相机渲染是否仍系统性空白。224×400 只做必要的前反向和渲染接口检查。短程检查用于发现结构性失败，不尝试多组配置、不按 mini 分数挑方案；其结果单独标注为调试 / 短程结果。若仍出现与首轮相同的系统性空渲染，或深度头整幅常数，则停止这一适配尝试并记录，不自动进入长训练。
4. **通过流程检查后按既定预算正式实验。** 两个分辨率均维持 batch size 1、100001 步、学习率 `2e-4`、骨干倍率 `0.1`、1000 步 warm-up、每 1000 步 10 个样本 val、每 10 次 val 完整 mini、训练结束强制 mini；最终各做完整 total。两种分辨率共享同一适配方式，不分别调参。正式训练仍从 VGGT 初始化开始，调试检查点不自动混入正式实验。
5. **如实汇报结果。** 保留 `all_18`、`novel_12` 和诊断用 `input_6` 的四项指标、参数量与完整重建耗时。PCC 未定义时继续记录数量及原始状态，不删除样本、不替换为更好看的均值；质量一般也报告，不追加优化轮次或模型变体。

固定 mini 是现有 2048-bin 子集；短程检查与正式结果必须使用不同目录及完整的初始化 / 步数记录，不能把短程检查分数当成最终对比结果。正式产物仍放到软链接后的 `work_dirs`，临时产物全部放 `/tmp`。

### 13.4 必要的可观测性补充

每 1000 步的 10 样本 val 补充保存输入 / 渲染 RGB、学生深度头、渲染深度和 alpha，并保存原始深度数组及每幅图的最小值、最大值、标准差。深度图使用同一样本内可比较的色标，同时保留原值，防止逐图自动拉伸掩盖常数输出；将“深度头整幅常数”与“alpha 为零导致的零深度渲染”分别记录。这仅用于诊断，不引入新的损失或数据过滤。

W&B 继续离线。飞书推送失败作为独立日志集成问题修复，不能通过推送失败停止训练，也不能以后台写日志替代助手监控。本节为确认时的方案；后续实现和检查进度以第 14 节为准。

## 14. 已知输入相机适配的实现与检查（2026-10-03）

### 14.1 已实现的数据流与实验隔离

新增 `omniscene_knowncam_112x200` 和 `omniscene_knowncam_224x400` 两套实验，继承原 `omniscene_common`。模型使用 `model.encoder.camera_source: known`，数据使用 `load_input_cameras: true`，评估使用 `test.camera_alignment: known_input_frame`。原两套 RGB-only 配置继续使用默认 `camera_source: predicted` 和输入六相机 Sim(3)。原模型结构、参数量、损失、优化器、正式步数及 val / mini 节奏均未修改。

已知相机模式在训练 / val 中只额外加载六路输入标定。`src/geometry/omniscene_cameras.py` 实现第 13 节的固定坐标变换，协议版本为 `first_input_mean_center_radius_v1`；坐标计算关闭混合精度，以避免 bf16 改变标定。encoder 用裁剪后的像素内参和规范化 w2c 反投影学生深度；训练渲染使用对应 c2w 和归一化内参。`pred_context_pose` 始终表示网络预测，实际用于构建和渲染的相机另存为 `construction_pose`，输入相机确定的坐标变换另存为 `camera_transform`。

mini / total 在高斯构建完成后，将目标 18 路相机应用同一个 `camera_transform`，不再在新模式下使用预测相机进行 Sim(3)。RGB 骨干只处理六路输入图像；目标相机和目标 RGB 不进入 encoder。网络的预测相机头及 VGGT 相机蒸馏保持原样。

检查点记录相机来源和规范化版本。训练续训及独立评估都会检查协议，阻止 RGB-only / 已知相机检查点混用；旧 RGB-only 检查点缺少新增字段时按原模式兼容。这里检查的是配置和检查点协议，不审查数据集。新实验使用独立目录，首轮产物保留在数据盘软链接对应的原目录。

重建计时包含新模式所需的输入内外参传输（含 H2D 的口径）、输入相机规范化及完整高斯构建；目标相机处理、渲染、指标和文件输出仍在重建计时之外。`run_metadata.json`、`evaluation_summary.json` 记录相机模式，便于区分比较条件。

### 14.2 可视化与通知

每轮 val 的 10 个样本保留原 RGB 对比 PNG，同时在同名子目录保存：

- `rgb.png`：输入 / 渲染 RGB，上下两行、各六路。
- `depth.png`：学生深度 / 渲染深度，上下两行、各六路，共用该样本的深度色标。
- `alpha.png`：六路渲染 alpha。
- `depth_head.npy`、`rendered_depth.npy`、`alpha.npy`：原始浮点数组。
- `statistics.json`：逐视角最小值、最大值、均值、标准差、是否整幅常数、是否全零、有限值比例、色标范围，以及预测 / 构建相机的分别记录。

mini / total 额外保存 `depth_diagnostics.json`，记录全部 bin 的深度头、渲染深度及 alpha 统计；不会据此过滤样本或改变指标。常数深度判断针对整幅图的实际数值，空渲染通过 alpha 全零单独记录。

飞书模块导入失败的已确认原因是 `anysplat` 环境缺少 `python-dotenv`。已加入依赖并在当前环境安装，导入检查通过；修正了 `send_feishu` 返回 False 时子进程仍成功退出的问题。通知异常仍只记录日志，不中断训练。调试期间关闭飞书，尚未以真实消息验证送达。

### 14.3 正式运行入口

从项目根目录、在 `anysplat` 环境中执行：

```bash
# 两种分辨率使用独立工作目录，从 VGGT 初始化；同一目录重启时自动续训。
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=omniscene_knowncam_112x200
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=omniscene_knowncam_224x400

# 已知相机模式的自训模型评估：以 112×200 为例。
CUDA_VISIBLE_DEVICES=0 python -m src.main \
  +experiment=omniscene_knowncam_112x200 mode=test test.split=total \
  checkpointing.load=work_dirs/omniscene_knowncam_112x200/checkpoints/final.ckpt \
  hydra.run.dir=work_dirs/omniscene_knowncam_112x200/eval/total

# 作者预训练模型仍沿用原 RGB-only 条件。
CUDA_VISIBLE_DEVICES=0 python -m src.main \
  +experiment=omniscene_112x200 mode=test test.split=total \
  test.weights_source=author test.pretrained_path=pretrained \
  hydra.run.dir=work_dirs/author_112x200/eval/total
```

224×400 评估替换相应实验名和路径；mini 评估将 `test.split=total` 与目录末尾的 `total` 均改为 `mini`。正式命令不继承 `/tmp` 的短程检查点。


### 14.4 实际检查结果与停止边界

所有调试产物位于 `/tmp/anysplat-knowncam-20261003/`；依赖 / 模型缓存沿用 `/tmp/anysplat-omniscene-debug-Qtfm6h/`，pip 缓存位于 `/tmp/anysplat-pip-cache/`。未向正式 `work_dirs` 写入新实验，也没有重启首轮训练。以下不是 100001 步完成后的正式实验结果。

- **CPU 回归：21 项全部通过。** 包括原协议、已知相机两套配置、只加载输入 RGB 与输入标定、两种裁剪分辨率的反投影再投影、带非零旋转 / 平移 / 尺度的坐标变换、混合精度下标定精度、检查点协议隔离、常数深度与空渲染的分别记录、PCC 统计、通知失败不阻断，以及 Lightning 完整续训。
- **作者权重 RGB-only 回归通过。** 112×200 单 bin 的三组四项指标与改动前接口调试结果在数值精度范围内一致，最大绝对差约 `3.48e-7`；仍标记 `complete=false`，不构成作者权重完整 mini 结果。原检查点 62800 步的协议也已验证可供原配置读取、会被新配置拒绝误用。
- **224×400：两步真实训练与接口检查通过。** 从 VGGT 初始化，完成前反向、10 样本 val、完整检查点保存及单 bin 最终 mini。评估明确为 `complete=false`、1/2048，不作为完整 mini 成绩。10 个 val 样本共 60 路深度头 / 渲染深度均非常数、有限，未出现整幅 alpha 为零。
- **112×200：一次 1000 步短程训练与完整 mini 完成。** 从 VGGT 初始化，使用原损失和优化器；仅为控制调试产物将 checkpoint 间隔设为 1000、保留 1 个，训练 / 测试数据加载 worker 设为 2（val 保持 1），日志使用本地 CSV，飞书关闭；正式默认配置保持原值。1000 步 val 为 10 个样本，PSNR **27.1146 dB**、总损失 **0.0132051**。完整 mini 为 **2048/2048**，无 batch 截断，三个视角组共 6144 条指标记录。

| 1000 步 mini 视角组 | PSNR（dB） | SSIM | LPIPS | PCC |
| --- | ---: | ---: | ---: | ---: |
| `all_18` | 12.1948 | 0.4817 | 0.5341 | 0.1002 |
| `novel_12` | 5.1787 | 0.2885 | 0.7023 | 0.0608 |
| `input_6` | 26.2269 | 0.8682 | 0.1978 | 0.2952 |

三组的 PSNR、SSIM、LPIPS、PCC 汇总均有定义，没有排除任何未定义条目。PCC 按既定视角组整体展平计算：某些单视角的渲染深度为常数，不等于该 bin 的整个视角组 PCC 必然未定义。

完整 mini 的模型输出统计：

- 12288 路输入深度头：整幅常数 **0**，非有限值图像 **0**。
- 12288 路输入视角：整幅 alpha 为零 **0**。
- 24576 路新视角：整幅 alpha 为零 **17087**，即 **69.53%**；这些视角对应零渲染深度。36864 路总渲染深度中的常数图也恰为这 17087 路。
- 1000 步 val 的 60 路输入深度头与渲染深度均非常数、有限，整幅 alpha 为零 **0**。

从保存的 1000 步检查点独立加载，在与第 12 节相同的 mini 索引 0、300、1000 上复核：对同一份高斯，按构建相机单独渲染输入 6 路，与按正式 18 路评估流程渲染后取最后 6 路，其相机矩阵、RGB、渲染深度最大绝对差均为 **0**。三例输入 PSNR 分别为 26.7546、24.7525、25.5282 dB。这验证了新模式的构建 / 监督 / 评估相机一致；不代表真实三维结构已经准确。

参数量与首轮完全相同：学生可训练 886363720、冻结 304372736、合计 1190736456。此次完整 mini 排除前 5 个预热样本后，2043 个样本完整重建均值 **71.578 ms/bin**，含输入 RGB / K / c2w 传输的均值 **71.679 ms/bin**；RTX 4090，未强制独占，不能当作正式最终性能。

**结论与停止位置：** 最小适配已消除本次检查中的输入视角相机条件不一致，真实标定下输入 6 路可正常重建。但新视角仍只有 **5.1787 dB**，且空渲染比例高，尚未得到可视为有效新视角重建的结果。深度头没有整幅常数，不能把这次失败称为用户定义的深度退化。按“仅做最小适配、不继续调优”的边界，停在这一次短程检查；未加入 Metric3D 或新视角监督、未尝试其他损失 / 学习率 / 结构，未自动启动两套正式长训练。

完整证据在 `short112/eval/final-mini-step-00001000/`；val 可视化在 `short112/val/step-00001000/`；相机一致性复核在 `camera_check/summary.json`；统一核对摘要为 `checked_results.json`。以上路径均相对于 `/tmp/anysplat-knowncam-20261003/`，属于临时调试证据，不保证永久保留，关键结论已转录到本文。
