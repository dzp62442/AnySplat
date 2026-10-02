# AnySplat：OmniScene 数据集实验文档

本文记录已确认的实验方案及其代码实现。两套配置、训练、自动续训、mini/total 评估和作者权重评估均已接入。实现与调试说明见第 11 节；没有启动正式训练或完整数据集评估，也没有审查或重新预处理数据集。

核查日期：2026-10-02。代码依据如下，结论以实际代码为准：

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
