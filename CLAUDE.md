# SSM × VLM 风险评分 benchmark（博士论文）

## 这个项目在做什么
- 用自车摄像头图像，让 VLM（视觉大模型）给每张图的风险单独打分。
- 用标注好的 3D 轨迹，独立算出多种替代安全指标（SSM）。
- 两边都转成排名 / 百分位后比较，看 VLM 理解的"风险"最像哪个 SSM。
- 关注普通驾驶场景，不是碰撞或极端场景；也不是车道偏离研究。

## 协作约定（给 Claude）
- 用简体中文、平实的话回答；代码、文件名、论文原文用英文。
- 公式不用 LaTeX，用纯文本或代码块，并用文字解释公式的意思。
- 改研究设计（指标定义、参数、配对规则、筛选条件）之前，先列出选项并问我。
- 下载任何文件前，先说清文件名、来源、大小，等我同意。
- 指标对照表只有一份：根目录的 `SSM_dataset_metric_matrix.xlsx`，不要另存副本。
- 每做完一个重要决定，更新本文件，再 commit + push，让另一台电脑也能看到。开始工作前先 pull。

## 仓库里有什么
- `SSM_dataset_metric_matrix.xlsx`：23 个 SSM × 19 个数据集的对照表；另有按指标类型、按碰撞类型分类的公式表，以及 △ 条件说明。
- `code/`：Waymo 处理流程（见下）。
- `labels/`：人工标注，无法重新生成，必须进仓库。
- `docs/ssm_computation_steps.md`：每个 SSM 的详细计算步骤（已实现的和程序一致，未实现的是建议做法）。改了算法就同步改这份文档。
- `results/`：运行结果，不进仓库，可以重新生成。

## 数据（不进仓库）
- 只用 Waymo Open Dataset Perception **v1.4.3**（带地图），不混用 v2.0.1。除了地图，两个版本内容相同。
- 下载：在 waymo.com/open 注册后，用 Google Cloud SDK（gcloud）登录同一个 Google 账号，从 `gs://waymo_open_dataset_v_1_4_3/individual_files/validation/` 下载 `segment-<片段名>_with_camera_labels.tfrecord`。每段约 1 GB，验证集 202 段共约 192 GiB。
- 目前只下了 3 个 demo 片段：
  - `16229547658178627464_380_000_400_000`：旧金山，白天，行人多
  - `14486517341017504003_3406_349_3426_349`：凤凰城，白天，约 15 m/s
  - `13184115878756336167_1354_000_1374_000`：夜间
- 办公室电脑：原始文件在 `F:\benchmark\waymo perception\v1.4.3_tfrecord\validation\`，提取结果在 `F:\benchmark\waymo perception\extracted\validation\`（F 盘是 U 盘，可以带走）。
- 其他电脑把自己的数据路径写进本机的 `CLAUDE.local.md`（不进仓库）。

## 流程（`code/`，Windows 上用普通 Python 即可，不需要 TensorFlow）
依赖：numpy、pandas、pyarrow、protobuf、scipy、matplotlib、Pillow。

1. 提取：`python -I code/wod_v143_extract.py <tfrecord 文件夹> <提取输出文件夹>`
   每段输出 frames / objects / projected_boxes / camera_boxes / map 五个 parquet、context.json 和 5 个相机的图像；格式见脚本开头的说明。
2. 检查：`python -I code/check_v143_extract.py <提取输出文件夹> <结果文件夹>/checks`
3. 逐帧算 SSM：`python -I code/waymo_ssm_frames.py <提取输出文件夹> <结果文件夹> [--camera FRONT|all]`
   输出 frame_ssm.csv（逐帧）、scene_summary.csv（逐段）、params.txt、TTC 对比图和快照图。

参数在 `code/waymo_ssm_frames.py` 顶部的 PARAMS，几何假设在 `code/wod_common.py` 顶部。

## 已经定下来的做法
- 比较方式：VLM 分数和 SSM 都转成排名 / 百分位。主指标用 Kendall τ-b，另报 Spearman（并列取平均秩）和 NHTSA 的成对一致率（arXiv 2306.14657），用配对 bootstrap 判断哪个 SSM 更接近。
- 并列多的 SSM（TTC 为无穷大、PET 不存在）用两步法：先看 VLM 能不能分出有无冲突（AUC），再只在有值的帧里比排名。
- TA、CS 已删除：要先判断"谁在避让"，轨迹数据看不出来。表里现在 23 个 SSM。
- 只算 FRONT 相机里最清楚的物体（`--camera FRONT`），和 VLM 看到的画面一致。
- 前车配对用地图车道路线法（`lead_map`）；自车不在车道上时，退回"沿自车实际路线"法。直线走廊法只作对照。
- 前车的定义：同车道、同方向、前方最近的车。同车道 = 车身中心离车道中心线 < 1.75 m；偏离车道中心（压线、正往外变道）的车不算前车。
- Waymo 的 3D 标注只到约 80 m。更远的前车数据里没有，按"没有前车"处理；但 VLM 看图时可能看到它，论文里要说明。
- 标注里的速度和加速度在全局坐标系（已验证）。
- 参数默认值（demo 用，最终由我决定）：
  - 反应时间 1.0 s，敏感性分析用 0.7 / 1.5 / 2.2 s
  - 急刹减速度 6 m/s²，敏感性分析用 3.4 / 8.5 m/s²
  - 路面摩擦系数 μ = 0.7；DST 安全时距 1.4 s；TET/TIT 的 TTC 阈值 3 s
  - RSS：反应期间加速 3.8、后车刹车 6、前车最大刹车 7 m/s²，反应时间同上

## 参数的文献依据
- 反应时间：SHRP2 碰撞和险情平均约 1.3 s、90% 分位 2.2 s（NCHRP 15-75）；Green 2000：有准备约 0.7 s、看到刹车灯约 1.25 s、受惊吓约 1.5 s；AASHTO 设计值 2.5 s。
- 减速度：AASHTO 设计值约 3.4 m/s²（偏舒适）；SHRP2 险情平均约 6.7 m/s²；车辆最大可用减速度平均 8.45 m/s²（Cunto & Saccomanno 2008）。
- 各 SSM 的参数取值汇总见 Singh et al. 2023（arXiv 2306.14657）。

## 已知问题 / 待决定
- ACT：约一半的帧 ACT < 1.5 s，几乎都来自对向车道正常会车的车（横向约 3 m）。要定一个筛选方式，例如只算可能相撞的物体，或用地图排除对向车道。
- 自车外框：假设坐标原点在后轴，车头往前 4.0 m、车尾往后 1.1 m、半宽 1.0 m，尚未核实。
- 2D-TTC：目前用几何版（两车外框第一次重叠）；Guo et al. 2023 还有纵向、横向分开算再取最小的版本，未定。
- CI（Ozbay 2008）的公式还没对照原文。
- VLM 看哪几个相机、怎么打分（比如 0–100、是否多次采样）还没定。

## 前车配对的人工检查（2026-10-14，3 个 demo 片段）
- 标了 93 帧：三种方法选得不一样的 43 帧全部标了，另从选得一样的帧里随机抽了 50 帧。标注时看不到是哪种方法选的。
- 文件：`labels/waymo_demo_lead_labels.csv`（answer 是原始答案，answer_final 是按规则修正后的答案，change 写了修正原因）、`labels/waymo_demo_lead_key.csv`。
- 脚本：`code/make_lead_labeling.py` 生成标注页面，`code/score_lead_labels.py` 计分。
- 结果（80 m 外的前车按"没有前车"算）：
  - 地图法：不一致的帧对 41/43，全部 595 帧估计 99.7%
  - 沿路线法：16/43，估计 95.5%
  - 直线法：10/43，估计 94.5%
- 地图法还错的 2 帧：前车开始驶离本车道，但离车道中心才 0.8 m。
- 标注页面的问题：远处两个框叠在一起时，编号容易看错（有 3 帧因此看错，已经修正）。

## 下一步
- 扩到验证集 202 段：边下载边提取，原始文件不存。
- VLM 打分，然后和 SSM 做排名比较。
- 家里电脑：用另一个数据集测试。只要写一个转换脚本，输出和上面相同格式的每段文件，`wod_common.py` 和 `waymo_ssm_frames.py` 就能直接用。
