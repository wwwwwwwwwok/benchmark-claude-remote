# 一次性检查脚本（存档）

2026-10-14 到 10-16 讨论参数和算法时写的检查脚本。原来放在 Claude 桌面版会话的临时目录里，那里会被清掉，所以存到这里。它们不是主流程的一部分。

- 用法大多是 `python -I <脚本> <仓库的 code 文件夹> <提取输出文件夹> [frame_ssm.csv]`，具体看每个文件开头的说明。
- 脚本是按当时的代码写的，后来主程序又改过（比如 77 m 规则、ACT 直接投影）。现在重新运行，数字可能和当时略有不同；结论以 `CLAUDE.md` 和 `docs/ssm_computation_steps.md` 里记的为准。
- `delta_v_check.py` 里写死了办公室电脑的路径。

| 脚本 | 回答的问题 | 结论 | 记在哪里 |
|---|---|---|---|
| `label_accel_match.py`、`label_accel_match2.py`、`label_accel_match3.py` | Waymo 标注里的速度、加速度是怎么算出来的 | 对框中心位置做 Savitzky–Golay 求导，19 帧（1.9 s），二次多项式。在世界坐标里（不加地图偏移，`_match3`）速度误差 0.003 m/s、加速度误差 0.006 m/s²；加了地图偏移（`_match2`）误差约 0.03 | CLAUDE.md "自车速度和加速度"；docs 0.3 |
| `label_accel_smoothness.py` | 标注的加速度有多平滑（Punzo et al. 2011 的 jerk 正负翻转比例） | 34%，和自车用 1.5 s 窗口差不多（33%） | docs 0.3 |
| `accel_window_check.py`、`accel_window_criteria.py` | 自车加速度的平滑窗口取多大；按文献的准则（Thiemann et al. 2008、Punzo et al. 2011）比较 | 最后定为和标注同一方法（SG 19 帧） | docs 0.3 |
| `ego_19.py` | 自车用 SG 1.9 s 对位置求导，和对 Waymo 给的速度求导比较 | 用位置 | CLAUDE.md "自车速度和加速度" |
| `speed_conv.py` | 速度大小 vs 沿车道方向的速度分量 | 接近速度最多差 0.07 m/s，有 TTC 的帧相同，排名一致程度 0.999。后来按交通安全的标准做法用速度大小 | CLAUDE.md "速度和加速度" |
| `act_course_check.py` | ACT 原文 Eq. 15 的"可能相撞"条件有多大影响 | ACT < 1.5 s 的帧从 75% 降到 1% | CLAUDE.md "ACT"；docs 第 4 节 |
| `exp_act_analytic.py` | ACT 的接近速度：0.1 s 差分 vs 直接投影 | 排名一致程度 0.9993，改成直接投影 | 同上 |
| `exp_lowspeed_ttc2d.py`、`exp_lowspeed_ttc2d_75.py` | 静止物体的速度噪声，"在动"门槛 0.1 还是 0.5 m/s；精确 2D-TTC vs 每 0.05 s 查一次、只查到 10 s；75 m 以外有哪些二维指标的对象（`_75` 按 75 m 查） | 2D-TTC 改成精确算法；门槛不用 0.1 | CLAUDE.md "已知问题"里的 2D-TTC；docs 第 8 节 |
| `exp_ped_ttc.py`、`exp_ped_thr.py`、`exp_ped_thr02.py` | 行人速度的分布；行人门槛 0.3 和 0.2 m/s 比较（`_thr` 比 0.5 和 0.3，`_thr02` 比 0.3 和 0.2）；同一辆前车的 2D-TTC 和 TTC | 行人 0.3 m/s，车、骑车人、自车 0.5 m/s | CLAUDE.md "在动的门槛"；docs 第 8 节 |
| `exp_label_edge.py`、`exp_label_edge2.py` | Waymo 的 3D 标注到多远为止 | 边界是"框上离自车最近的点 77 m" | CLAUDE.md "Waymo 的 3D 标注范围"；docs 0.5 |
| `exp_edge_accel.py`、`exp_edge_speed.py` | 轨迹头尾的加速度、速度有多不准 | 最边上一帧加速度误差中位数约 0.2 m/s²，速度约 0.1 m/s；往里 9 帧以后为 0。头尾帧做标记，主分析去掉 | docs 0.10 |
| `exp_lead_params.py` | 找前车的同方向阈值 30° 和车道匹配参数，改大改小有没有影响 | 前车一帧都没变 | docs 8.3 |
| `exp_pet_feasibility.py` | PET 能不能做 | 3 段共 60 s 里真正交叉的只有 1 对，删掉 PET | docs 第 7 节 |
| `delta_v_check.py` | delta-V 在普通场景里的表现 | 车对车时和接近速度排名完全一样；扩展版全是 0，删掉 | docs 第 7 节 |
| `seg_label_check.py` | 有多少帧带激光雷达语义标签、相机全景分割标签 | 每段约 30 帧有激光雷达语义标签；demo 里没有相机全景分割标签 | CLAUDE.md "车型" |
| `pdf2txt_cols.py`、`render_pages.py` | 读论文 PDF 的小工具：双栏 PDF 转文字；把某几页渲染成图片（公式在文字里会乱，要看图） | — | — |
| `scan_workbook.py`、`remove_pet_gt_mprism_workbook.py` | 改指标对照表的模板：先查一个指标出现在哪些单元格；再删行，同时移动合并单元格、行高、筛选范围，改 COUNTIF 的范围 | 用之前先关掉 Excel，否则保存会失败 | — |
