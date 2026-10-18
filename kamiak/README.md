# 在 Kamiak 上跑 VLM（开源 Qwen）

思路：在 GPU 节点上用 vLLM 把开源的 Qwen 模型启动成一个本地服务，再用 `code/run_vlm.py` 把每个目标帧的图和问题发过去。不按量收费，模型版本固定，可以复现，还能拿到每个回答的概率。

## Kamiak 上相关的资源（2026-10 查的官方页面，用前在集群上再确认）
- `camas` 分区：3 个节点，每个 4 块 H100 80GB，所有研究人员都能用，GPU 任务不会被抢占。
- `kamiak` 分区（共享）：也有 H100 和 A100 节点，但会被出资课题组的任务抢占；我们的程序能断点续跑，被抢占了重新提交就行。
- `vcea` 分区：2 块 A100，只给工学院（VCEA）的人用。
- 存储：家目录 100 GB；课题组目录 `/data/lab/<组名>` 每个 PI 500 GB；临时目录 `/scratch/user/<netid>/`（用 `mkworkspace` 建，每个最多 10 TB，2 周后自动删除）。
- 模型大小：Qwen3-VL-32B（BF16）约 66 GB，放课题组目录；用 2 块 H100 跑比较宽裕。

## 一次性准备（这些步骤要你自己在 Kamiak 上操作，账号和密码都不经过我）
1. 登录：`ssh <netid>@kamiak.wsu.edu`。
2. 把仓库 clone 到课题组目录或 scratch（仓库是私有的，要在 Kamiak 上配好 GitHub 的 SSH key 或 token）。
3. 建 Python 环境并装 vLLM（以后版本可能不同，按 vLLM 官方说明）：
   ```bash
   module avail python          # 选一个 3.10 以上的版本 module load
   python -m venv /data/lab/<组名>/venvs/vllm
   source /data/lab/<组名>/venvs/vllm/bin/activate
   pip install vllm openai pandas pillow
   ```
4. 下载模型权重（约 66 GB）：
   ```bash
   pip install -U huggingface_hub
   huggingface-cli download Qwen/Qwen3-VL-32B-Instruct --local-dir /data/lab/<组名>/models/Qwen3-VL-32B-Instruct
   ```
5. 把 VLM 输入图片传上去（图片不在 git 里）。在办公室电脑上，仓库根目录下运行：
   ```bash
   rsync -av results/vlm_inputs results/vlm_inputs_separate <netid>@kamiak.wsu.edu:<Kamiak 上仓库的路径>/results/
   ```
6. 打开 `kamiak/qwen_vlm.sbatch`，把开头标了 `CHANGE_ME` 的路径改成你的；标了 `CHECK` 的几行（分区、GPU 写法、时限）用 `sinfo -p camas`、`scontrol show partition camas` 核对一下。

## 运行
```bash
cd <Kamiak 上仓库的路径>
python -I code/run_vlm.py --grid_targets results/vlm_inputs/targets.csv \
    --sep_targets results/vlm_inputs_separate/targets.csv --out results/vlm_scores/check.jsonl --show 1
# ↑ 只检查、不发送：打印一个组装好的请求和总规模（demo：1,530 次请求）
sbatch kamiak/qwen_vlm.sbatch
squeue -u $USER                 # 看任务状态；日志在 logs/
```
- 结果在 `results/vlm_scores/<RUN>.jsonl`，每行一个请求：分数、危险来源、理由、token 用量、耗时和回答的概率。
- 任务中断（时限到了、被抢占）后，再提交一次就会从没做完的地方接着跑。
- 跑完把结果传回办公室电脑：`rsync -av <netid>@kamiak.wsu.edu:<路径>/results/vlm_scores ./results/`。
