# 在 Kamiak 上跑 VLM（开源 Qwen）

思路：在 GPU 节点上用 vLLM 把开源的 Qwen 模型启动成一个本地服务，再用 `code/run_vlm.py` 把每个目标帧的图和问题发过去。不按量收费，模型版本固定，可以复现，还能拿到每个回答的概率。

跑过三个模型（都是 2026-10-19，用户定的）：
- Qwen3.6-27B（保留 thinking）；
- Qwen3-VL-32B-Instruct（稠密，不思考）；
- Qwen3-VL-32B-Thinking（和 Instruct 同一个底座、同样大小，先思考再回答；和 Instruct 一比，就能单独看出思考的作用）。

用哪个由 sbatch 里的 `MODEL` 决定，默认是 `qwen3-vl-32b`。下面的路径都是这个账号的（课题组 `cejli`，个人目录 `sy`）。

## Kamiak 上的规矩和资源（2026-10-19 在集群上查过）
- 登录节点上不能跑程序、装包，也不能长时间传文件。官方原话："Do not run applications or installs on the login nodes, use sbatch or idev to run them on a compute node."
  - 装包：`srun --partition=kamiak ...`（默认的共享分区，用 CPU 就够）。
  - 下大文件：`srun --partition=dtn ...`（专门传数据的节点）。
  - 跑模型：`sbatch`。不要用挂在终端（或 Claude 会话）上的 `srun` 跑长任务：终端一断，作业就被取消（2026-10-19 出过一次）。
- GPU 驱动：全集群统一是 535.274.02，最高支持 CUDA 12.2（查过 camas 的 sn17、sn18，vcea 的 sn14，kamiak 分区的 sn19）。
  - 所以只能用 CUDA 12.x 编译的 PyTorch 和 vLLM。直接 `pip install vllm` 装的是 CUDA 13.0 版，在这里跑不起来。
  - 不要 `module load cuda`，pip 包自带运行库。
- `camas` 分区：sn16–sn18，每台 4 块 H100 PCIe 80GB，所有研究人员都能用，GPU 任务不会被抢占。
  - 三台经常都有人在用，可能要排队。
  - GRES 的名字是 `tesla`，写 `--gres=gpu:2` 就行。时限最长 7 天。
- `kamiak` 分区（共享）：也有 GPU 节点（比如 sn19，2 块 H100 NVL 94GB），但会被出资课题组的任务抢占。我们的程序能断点续跑，被抢占了重新提交就行。
- `vcea` 分区：sn14 有 2 块 A100 40GB，`cejli` 在允许名单里。
- `dtn` 分区：只有 dtn1 一台，8 核，内存 31,835 MB，其中 6,000 MB 留给系统。
  - 常有别人在用，内存要申请得很少（≤900M），否则会一直排队。
  - dtn1 上没有 `/opt/apps`，不能 `conda activate`；系统 Python 是 3.9。
- 存储：
  - 家目录 100 GB（2026-10-19 只剩 13 GB）。
  - 课题组目录 `/data/lab/cejli`，个人的东西放 `/data/lab/cejli/sy`。
  - 临时目录 `/scratch/user/<netid>/`（用 `mkworkspace` 建，每个最多 10 TB，2 周后自动删除）。
  - `/tmp` 是每台机器自己的，别的节点看不到，脚本和日志不要放在 `/tmp`。
- 模型（都放在 `/data/lab/cejli/sy/models/`，下载后都按官方校验值逐个核对过）：
  - `Qwen3.6-27B`：15 个分片，51.75 GiB。一块 H100 就装得下（权重占显存 51.01 GiB）。
  - `Qwen3-VL-32B-Instruct`：版本 `0cfaf48183f594c314753d30a4c4974bc75f3ccb`，14 个分片，66,726,519,322 字节（62.14 GiB）。要 2 块 H100。
  - `Qwen3-VL-32B-Thinking`：版本 `7edd10ffd1196091948fb245ff63e406ccb2d4d1`，14 个分片，66,726,519,215 字节（62.14 GiB）。要 2 块 H100。
  - 原来 `hf_cache` 里的 Qwen3-VL-30B-A3B-Instruct（MoE）2026-10-19 按用户要求删掉了；删之前核对过和官方一致，要用可以重新下。
  - 脚本默认用 2 块卡。

## 一次性准备（2026-10-19 在这个账号上已经做完，下面记的是实际能用的做法）
账号、密码、GitHub token 都由你自己设置，不经过 Claude。

1. 仓库：`git clone https://github.com/wwwwwwwwwok/benchmark-claude-remote.git ~/benchmark-claude-remote`。
   - 仓库现在是公开的。push 要 GitHub token：先 `git config --global credential.helper store`，第一次 push 时输入一次，以后就记住了。
2. 建 conda 环境，装 CUDA 12.9 版的 vLLM（在计算节点上装）：
   ```bash
   srun --partition=kamiak --account=cejli --time=00:30:00 --cpus-per-task=4 --mem=16G bash -c '
   source /opt/apps/miniconda3/3.12/etc/profile.d/conda.sh
   conda create -p /data/lab/cejli/sy/conda_envs/vllm_cu129 python=3.11 -y
   conda activate /data/lab/cejli/sy/conda_envs/vllm_cu129
   pip install "https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
       --extra-index-url https://download.pytorch.org/whl/cu129
   pip install openai pandas pillow
   pip install torchcodec --index-url https://download.pytorch.org/whl/cu129 --force-reinstall --no-deps
   python -c "import torch, vllm; print(torch.__version__, torch.version.cuda, vllm.__version__)"'
   ```
   - 最后一行应该打印 `2.13.0+cu129 12.9 0.32.0`。
   - vLLM 0.32.0 在 GitHub releases 上只有 cu129 这一个 CUDA 版本的 wheel，没有 cu128。
   - `torchcodec` 那一行不能省：pip 默认装的 0.17.0 和 cu129 的 torch 不配套，vLLM 启动时会报 `OSError: Could not load this library: .../libtorchcodec_image.so`。cu129 索引上最新的是 0.16.0+cu129。
3. 下载模型权重，在 dtn 上下。下面是 Qwen3.6-27B；另两个把仓库名、文件夹名换掉，再加上 `--revision`：
   - Qwen3-VL-32B-Instruct：`--revision 0cfaf48183f594c314753d30a4c4974bc75f3ccb`，2026-10-19 用了 9 分 25 秒；
   - Qwen3-VL-32B-Thinking：`--revision 7edd10ffd1196091948fb245ff63e406ccb2d4d1`，2026-10-19 用了 14 分 41 秒。

   ```bash
   srun --partition=dtn --account=cejli --time=03:00:00 --cpus-per-task=2 --mem=900M bash -c '
   export HF_HOME=/data/lab/cejli/sy/hf_cache
   /data/lab/cejli/sy/conda_envs/vllm_cu129/bin/hf download Qwen/Qwen3.6-27B --local-dir /data/lab/cejli/sy/models/Qwen3.6-27B'
   ```
   - 下完核对：用 HuggingFace API（`https://huggingface.co/api/models/<仓库名>/tree/<版本>`）拿到每个文件的官方 SHA256 / git 校验值，和本地重新算的比。2026-10-19 三个模型都逐个核对过，全部一致。
   - 命令是 `hf`：新版 huggingface_hub 把 `huggingface-cli` 改名了。
   - 要用 conda 环境里 `hf` 的完整路径（环境里是 huggingface_hub 2.2.0）。dtn1 上不能 `conda activate`；系统 Python 3.9 最多只能装到 huggingface_hub 1.8.0，用它下 Qwen3.6 会误报 "Model 'Qwen/Qwen3.6-27B' not found"。
   - `--mem` 不要超过 900M（见上面 dtn 的说明）。2026-10-19 实际下载用了 16 分 48 秒。
4. 把 VLM 输入图片传上去（图片不在 git 里）：
   - 先在 Kamiak 上建文件夹：`mkdir -p ~/benchmark-claude-remote/results`。图片一定要放进仓库的 `results/` 下面，因为 `run_vlm.py` 用的是相对路径。
   - 再在办公室电脑（Windows，没有 rsync）的仓库根目录下运行：
   ```bash
   scp -r results/vlm_inputs results/vlm_inputs_separate <netid>@kamiak.wsu.edu:~/benchmark-claude-remote/results/
   ```
   - demo 的大小：`vlm_inputs` 114 张图、31,516,386 字节；`vlm_inputs_separate` 570 张图、64,908,457 字节；各有一个 102 行的 `targets.csv`。传完用 `du -sb` 核对字节数（2026-10-19 核对过，一致）。
5. `kamiak/qwen_vlm.sbatch` 里的路径已经改成这个账号的。每个模型的设置写在脚本里 `case "$MODEL"` 那一段；加新模型就在那里加一段。

## 运行（在仓库根目录 `~/benchmark-claude-remote` 下）
```bash
/data/lab/cejli/sy/conda_envs/vllm_cu129/bin/python -I code/run_vlm.py --grid_targets results/vlm_inputs/targets.csv \
    --sep_targets results/vlm_inputs_separate/targets.csv --out results/vlm_scores/check.jsonl --repeats 5 --show 1
# ↑ 只检查、不发送：打印一个组装好的请求和总规模（demo 每帧问 5 次：2,550 次请求）。这一步很轻，可以在登录节点上跑
mkdir -p logs                   # Slurm 不会自己建日志文件夹；没有它，任务会直接失败而且没有日志
sbatch kamiak/qwen_vlm.sbatch --limit 2 --repeats 1 --out results/vlm_scores/qwen3-vl-32b/smoke_qwen3-vl-32b.jsonl
# ↑ 先小试：2 个目标帧 × 5 组 × 1 次 = 10 次请求
sbatch kamiak/qwen_vlm.sbatch   # 小试没问题再跑完整试跑（2,550 次请求），默认 MODEL=qwen3-vl-32b
sbatch --export=ALL,MODEL=qwen3-vl-32b-thinking kamiak/qwen_vlm.sbatch   # 换成别的模型（还有 qwen3.6-27b）
squeue -u $USER                 # 看任务状态；日志在 logs/
```
- 写在脚本名后面的参数会原样传给 `run_vlm.py`，并覆盖脚本里的同名参数。
- 三个模型共同的设置：同样的问题、5 个比较组、每帧问 5 次；同时发 32 个请求（`--workers 32`），时限 2 天；`VLLM_USE_FLASHINFER_SAMPLER=0`（原因见下面"出问题时先看这里"）。
- 每个模型自己的设置（采样都用模型自带 `generation_config.json` 里的推荐值，明确发给服务，也记进每行输出）：

  | | Qwen3-VL-32B-Instruct | Qwen3-VL-32B-Thinking | Qwen3.6-27B |
  |---|---|---|---|
  | `MODEL=` | `qwen3-vl-32b` | `qwen3-vl-32b-thinking` | `qwen3.6-27b` |
  | 采样 | 温度 0.7、top_p 0.8、top_k 20 | 温度 0.8、top_p 0.95、top_k 20 | 温度 1.0、top_p 0.95、top_k 20 |
  | 思考 | 没有 | 有，`--reasoning-parser qwen3` 把它分到 `reasoning` | 有，同左 |
  | `--max_tokens` | 1024（回答约 80 token） | 24576（思考也算在里面） | 24576（思考也算在里面；小试里最长 10,896） |
  | `--max-model-len` | 32768 | 40960 | 40960（最长的 sep 提示约 13,660 token，加上 24576 要装得下） |
  | 其他 | | | `--gdn-prefill-backend triton` |
  | 完整试跑用时（2,550 次） | 14 分 34 秒 | 1 小时 06 分 | 5 小时 52 分（另补跑 31 分 45 秒） |

- 不思考的模型千万不要加 `--reasoning-parser qwen3`：vLLM 0.32.0 的这个解析器默认把输出先当成思考内容，等不到 `</think>` 就会把整段回答放进 `reasoning`，`text` 是空的，全部解析失败。
- 结果在 `results/vlm_scores/<MODEL>/pilot_<MODEL>.jsonl`（小试是同一文件夹里的 `smoke_<MODEL>.jsonl`），每行一个请求，内容有：
  - 回答原文 `text` 和解析出的 JSON `parsed`；
  - 思考内容 `reasoning`，以及 `finish_reason`（是 `length` 就说明被 `--max_tokens` 截断了）；
  - 采样参数、token 用量（`usage`）、耗时（`latency_s`）；
  - 概率 `logprobs`：包括思考部分的 token，是温度、top_p、top_k 处理之前的原始概率。
- 小试要看四件事：
  - 服务起来没有：看 `logs/vllm_<任务号>.log`；
  - 10 次请求是不是都有回答，`parsed` 是不是都不为空；
  - `usage` 里的 `completion_tokens`，也就是思考加回答有多长，据此定 `--max_tokens`；
  - 每次要多久（`latency_s`），据此推算 2,550 次要多久、sbatch 的时限够不够。
- 思考到 `--max_tokens` 还没结束的回答（`finish_reason` 是 `length`，`text` 为空）用 Qwen 官方的"思考预算"做法补答（2026-10-20 用户定）：
  ```bash
  sbatch --export=ALL,MODEL=qwen3.6-27b kamiak/force_answer.sbatch            # 另一个：MODEL=qwen3-vl-32b-thinking
  ```
  - `code/force_answer.py` 把原来的思考原样接回去，加上 Qwen 文档里的收尾文字（"Considering the limited time by the user, I have to give the solution based on the thinking directly now."）并关上 `</think>`，让模型照已有的思考直接作答。还是原来那次抽样，不重新抽。
  - 服务不加 `--reasoning-parser`：回答里没有 `</think>`，加了解析器会把回答放进 `reasoning`。
  - 结果另存 `results/vlm_scores/<MODEL>/pilot_<MODEL>_forced_answers.jsonl`（每行有 `forced_answer: true`），原结果文件不动。
- 任务中断（时限到了、被抢占）后，再提交一次就会从没做完的地方接着跑。报错的请求不写进结果，下次会重发；回答解析不出来的会写进去（`parsed` 为空），不重发。
- 跑完把结果传回办公室电脑（在办公室电脑的仓库根目录下）：`scp -r <netid>@kamiak.wsu.edu:~/benchmark-claude-remote/results/vlm_scores ./results/`。

## 出问题时先看这里
- 任务马上失败、没有日志：多半是没有先建 `logs` 文件夹。
- `RuntimeError: The NVIDIA driver on your system is too old (found version 12020)`：装成了 CUDA 13.0 版的 PyTorch（`torch.version.cuda` 是 13.0）。按上面第 2 步装 cu129 版。
- `OSError: Could not load this library: .../torchcodec/libtorchcodec_image.so`：torchcodec 和 torch 不配套。按第 2 步从 cu129 索引重装。
- vLLM 日志里有 `FlashInfer kernels are disabled ... nvcc ... is missing`：这句可以不管。
  - FlashInfer 的内核要在第一次用时现场编译，但在 Kamiak 上它找不到 nvcc。H100 上 vLLM 默认还是会选 FlashInfer，到真正用的时候就可能出错。
  - 所以脚本用 `--gdn-prefill-backend triton` 和 `VLLM_USE_FLASHINFER_SAMPLER=0` 换成不用现场编译的写法。算的东西一样，可能稍慢。
- dtn 上的作业一直排队（`Reason=Resources`，`scontrol show job` 里预计开始时间很远）：内存申请多了，降到 900M 以下。
- dtn 上 `hf download` 报 "Model ... not found"：用的是系统 Python 3.9 的旧版 huggingface_hub。换成 conda 环境里 `hf` 的完整路径。
- `sbatch: error: Invalid generic resource (gres) specification`：GPU 的写法不对，按 `scontrol show node` 里 Gres 那一行改（camas 是 `gpu:tesla:4`）。
- 服务没起来：看 `logs/vllm_<任务号>.log`。显存不够可以把 `--max-model-len` 调小，或者用 4 块 GPU（`--gres` 和 `N_GPUS` 一起改）。
- `APITimeoutError: Request timed out.`：客户端等得太久。
  - openai 客户端默认每个请求等 600 s，超时后自动重试 2 次；每次重试都是重新抽一个回答，等于把思考长的回答换掉了。2026-10-19 的完整试跑因此有 98 条是重试后才答完的，另有 3 条失败。
  - 所以 `run_vlm.py` 现在每个请求最多等 1 小时（`--timeout 3600`），不自动重试。报错的请求不写进结果，重新提交时会补发。
  - `latency_s` 超过 600 s 的旧记录就是重试过的，已经移到 `results/vlm_scores/qwen3.6-27b/pilot_qwen3.6-27b_timeout_retried.jsonl`，并重新问过。
- 日志 `logs/qwen-vlm_<任务号>.out` 里，`run_vlm.py` 的进度行要等作业结束才一起出现（Python 写文件时有缓冲）。跑的过程中看进度，就数结果文件的行数：`wc -l results/vlm_scores/<MODEL>/pilot_<MODEL>.jsonl`。
- 作业显示 `CANCELLED by <你的 uid>`，内存也没超：多半是用终端里的 `srun` 跑的，终端（或 Claude 会话）断了。长任务用 `sbatch`。
- 报 `$'\r': command not found`：脚本带着 Windows 的换行符（比如是从办公室电脑直接拷过去的，不是 git clone 的）。用 `sed -i 's/\r$//' kamiak/qwen_vlm.sbatch` 修。
- 日志里说连不上 huggingface：模型已经在本地，在 sbatch 里 `conda activate` 那行后面加 `export HF_HUB_OFFLINE=1`。
