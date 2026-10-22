# 在 Kamiak 上跑 VLM（开源 Qwen、InternVL），以及下载、处理 Waymo 数据

Waymo 数据的部分见下面"Waymo 训练集 + 验证集"一节。

思路：在 GPU 节点上用 vLLM 把开源模型启动成一个本地服务，再用 `code/run_vlm.py` 把每个目标帧的图和问题发过去。不按量收费，模型版本固定，可以复现，还能拿到每个回答的概率。

跑过这些模型（都是用户定的）：
- Qwen3.6-27B（保留 thinking；2026-10-19）；
- Qwen3-VL-32B-Instruct（稠密，不思考；2026-10-19）；
- Qwen3-VL-32B-Thinking（和 Instruct 同一个底座、同样大小，先思考再回答；和 Instruct 一比，就能单独看出思考的作用；2026-10-19）；
- InternVL3.5-38B，同一个权重跑两次：不思考、思考（2026-10-20）。视觉部分是 InternViT-6B，语言部分是 Qwen3-32B。

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
  - 三台在集群登记的配置完全一样（`scontrol show node`：h100、hopper、每块 80GB、Platinum 8450Y+）。
  - **所有模型都在这里跑，每个作业 2 块 H100 PCIe 80GB**（2026-10-20 用户定：GPU 配置要一致）。不要为了不排队换到 sn19（H100 NVL 94GB）或 vcea（A100）。每个作业的日志开头会用 `nvidia-smi` 记下实际分到的显卡。
  - 每人同时最多用 2 块 GPU（第二个 2 卡作业会显示 `QOSMaxGRESPerUser`，等前一个结束）。
- `kamiak` 分区（共享）：也有 GPU 节点（比如 sn19，2 块 H100 NVL 94GB），但会被出资课题组的任务抢占。我们的程序能断点续跑，被抢占了重新提交就行。
- `vcea` 分区：sn14 有 2 块 A100 40GB，`cejli` 在允许名单里。
- `dtn` 分区：只有 dtn1 一台，8 核，内存 31,835 MB，其中 6,000 MB 留给系统。
  - 常有别人在用，内存要申请得很少（≤900M），否则会一直排队。
  - dtn1 上没有 `/opt/apps`，不能 `conda activate`；系统 Python 是 3.9。
- 存储：
  - 家目录 100 GB（2026-10-19 只剩 13 GB）。
  - 课题组目录 `/data/lab/cejli`，个人的东西放 `/data/lab/cejli/sy`。官方说每个导师默认 500 GB，更多要按年租（hpc.wsu.edu/users-guide/storage/，2026-10-21 查）。
  - 临时目录 `/scratch/user/<netid>/`：用 `mkworkspace` 建，每人一共最多 10 TB；每个最多存 2 周，到期连数据一起自动删，没有快照和备份。
  - `/tmp` 是每台机器自己的，别的节点看不到，脚本和日志不要放在 `/tmp`。
- 模型（都放在 `/data/lab/cejli/sy/models/`，下载后都按官方校验值逐个核对过）：
  - `Qwen3.6-27B`：15 个分片，51.75 GiB。一块 H100 就装得下（权重占显存 51.01 GiB）。
  - `Qwen3-VL-32B-Instruct`：版本 `0cfaf48183f594c314753d30a4c4974bc75f3ccb`，14 个分片，66,726,519,322 字节（62.14 GiB）。要 2 块 H100。
  - `Qwen3-VL-32B-Thinking`：版本 `7edd10ffd1196091948fb245ff63e406ccb2d4d1`，14 个分片，66,726,519,215 字节（62.14 GiB）。要 2 块 H100。
  - `InternVL3_5-38B`：OpenGVLab/InternVL3_5-38B 版本 `de99855be3642cd44fe97c9b72d70e5ce2c07f69`，36 个文件（16 个分片），76,797,097,400 字节（71.52 GiB）。要 2 块 H100（每块权重占 37.8 GiB，KV 缓存只剩 133,168 token）。
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
   - Qwen3-VL-32B-Thinking：`--revision 7edd10ffd1196091948fb245ff63e406ccb2d4d1`，2026-10-19 用了 14 分 41 秒；
   - InternVL3.5-38B：仓库 `OpenGVLab/InternVL3_5-38B`，`--revision de99855be3642cd44fe97c9b72d70e5ce2c07f69`，2026-10-20 用了 15 分 55 秒（用 `sbatch --wrap` 提交，会话断了也不影响）。

   ```bash
   srun --partition=dtn --account=cejli --time=03:00:00 --cpus-per-task=2 --mem=900M bash -c '
   export HF_HOME=/data/lab/cejli/sy/hf_cache
   /data/lab/cejli/sy/conda_envs/vllm_cu129/bin/hf download Qwen/Qwen3.6-27B --local-dir /data/lab/cejli/sy/models/Qwen3.6-27B'
   ```
   - 下完核对：用 HuggingFace API（`https://huggingface.co/api/models/<仓库名>/tree/<版本>`）拿到每个文件的官方 SHA256 / git 校验值，和本地重新算的比。2026-10-19 三个 Qwen、2026-10-20 InternVL3.5-38B 都逐个核对过，全部一致。
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
sbatch --export=ALL,MODEL=qwen3-vl-32b-thinking kamiak/qwen_vlm.sbatch   # 换成别的模型
#   MODEL 还可以是 qwen3.6-27b、internvl3.5-38b、internvl3.5-38b-thinking
squeue -u $USER                 # 看任务状态；日志在 logs/
```
- 写在脚本名后面的参数会原样传给 `run_vlm.py`，并覆盖脚本里的同名参数。
- 所有模型共同的设置：同样的问题、5 个比较组、每帧问 5 次；同时发 32 个请求（`--workers 32`），时限 2 天；`VLLM_USE_FLASHINFER_SAMPLER=0`（原因见下面"出问题时先看这里"）。
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

  InternVL3.5-38B（`generation_config.json` 里没有采样值，用的是用户选的值）：

  | | 不思考 | 思考 |
  |---|---|---|
  | `MODEL=` | `internvl3.5-38b` | `internvl3.5-38b-thinking` |
  | 采样 | 温度 0.8、top_p 0.8（模型页调用示例里的值），不设 top_k | 温度 0.6（模型页对思考模式的推荐），不设 top_p、top_k |
  | 思考 | 没有 | 问题和其他模型完全一样；`--chat-template kamiak/internvl_thinking_chat_template.jinja` 让回答以 `<think>` 开头（和 Qwen 思考模型的模板一样），`--reasoning-parser deepseek_r1` 把思考分到 `reasoning` |
  | `--max_tokens` | 1024 | 24576（vLLM 会自动截到剩下的上下文：sep 组只剩约 5,900） |
  | `--max-model-len` | 40960（它的语言模型最长就是 40,960） | 40960 |
  | 其他 | `--trust-remote-code` | `--trust-remote-code` |
  | 完整试跑用时（2,550 次） | 41 分 03 秒 | 1 小时 54 分 53 秒 |

- InternVL 要注意的地方：
  - `--trust-remote-code`：模型的配置类是随权重一起发的 .py 文件。2026-10-20 读过会被执行的两个（`configuration_internvl_chat.py`、`configuration_intern_vit.py`），只是存配置参数。
  - 图片按官方默认切成最多 12 块 448×448 加一张缩略图，每块 256 token，比 Qwen 多很多：每次请求 grid 约 14,400、sep 约 35,000、ctrl_grid 约 3,100、ctrl_sep 约 7,200 token（Qwen 是 7,250、13,650、1,650、2,900）。
  - 它的对话模板 vLLM 认成 `openai` 格式（和 Qwen 一样），每张图留在自己的说明文字后面，不会被挪到最前面。
  - 官方开启思考的办法是把系统提示词换成 `R1_SYSTEM_PROMPT`（`code/internvl_thinking_system_prompt.txt`，和模型页一字不差；`run_vlm.py --system_prefix_file` 可以把它拼在我们的系统提示词前面）。用户 2026-10-20 决定不用它，好让所有模型的问题一样。用它跑过一次小试作参照（`smoke_internvl3.5-38b-thinking_official_r1_prompt.jsonl`）。
  - 思考版有时把答案 JSON 写两遍，中间夹一个 `</think>`；有时在答案前后陷进"空格、换行"的死循环，一直写到上限。`run_vlm.py` 取回答里第一份 JSON（2026-10-20 用户定）。

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
- 写到 `--max_tokens` 还没读出答案的回答（`finish_reason` 是 `length`，`parsed` 为空）用 Qwen 官方的"思考预算"做法补答（2026-10-20 用户定）：
  ```bash
  sbatch --export=ALL,MODEL=qwen3.6-27b kamiak/force_answer.sbatch   # 还有 qwen3-vl-32b-thinking、internvl3.5-38b-thinking
  ```
  - 已经写出答案、之后才卡住的（InternVL 有 8 条）保留原答案，不补。
  - `code/force_answer.py` 把原来的思考原样接回去，加上 Qwen 文档里的收尾文字（"Considering the limited time by the user, I have to give the solution based on the thinking directly now."）并关上 `</think>`，让模型照已有的思考直接作答。还是原来那次抽样，不重新抽。
  - 服务不加 `--reasoning-parser`：回答里没有 `</think>`，加了解析器会把回答放进 `reasoning`。
  - 结果另存 `results/vlm_scores/<MODEL>/pilot_<MODEL>_forced_answers.jsonl`（每行有 `forced_answer: true`），原结果文件不动。
- 任务中断（时限到了、被抢占）后，再提交一次就会从没做完的地方接着跑。报错的请求不写进结果，下次会重发；回答解析不出来的会写进去（`parsed` 为空），不重发。
- 跑完把结果传回办公室电脑（在办公室电脑的仓库根目录下）：`scp -r <netid>@kamiak.wsu.edu:~/benchmark-claude-remote/results/vlm_scores ./results/`。

## Waymo 训练集 + 验证集：下载、提取、算 SSM（2026-10-21 写好；当天验证集跑完，训练集在跑）
用户 2026-10-21 定：在 Kamiak 上下训练集 798 段 + 验证集 202 段（测试集没有公开标注，不下），先把 SSM 全算出来，再决定 VLM 问哪些帧。

脚本：
- `kamiak/waymo_download.sbatch`：在 dtn 上从 Waymo 的存储桶下一个 split 到 scratch。下好的文件会跳过，所以中断了再交一次就接着下。gcloud 下载时会用存储桶里的校验值核对每个文件；下完脚本再核对每个文件的大小，全对才写 `DOWNLOAD_COMPLETE`。
- `kamiak/waymo_ssm.sbatch`：作业数组，一个任务管 10 段。
  - 先提取，用的是 `code/wod_v143_extract.py`，5 个相机都要，和办公室电脑一样。提取过的段会跳过。
  - 每段提取完，把表格（context.json 和 parquet，每段约 4.3 MB）复制一份到课题组目录。
  - 再对这 10 段跑 `check_v143_extract.py` 和 `waymo_ssm_frames.py`，设置和 demo 一样（5 个相机、自车按真实车身）。
- `code/merge_ssm_parts.py`：把各块的结果拼成和一次算完一样的文件。
  - 2026-10-21 在办公室电脑上用 demo 3 段测过：分块算、再拼起来，`frame_ssm.csv`、`scene_summary.csv`、`params.txt`、`extract_checks.csv` 和图都和一次算完的逐字节相同；`frame_ssm.csv` 也和 `results/waymo_demo/` 里试跑用的那份相同。
  - 有块缺了、有段重复、各块参数不一样时，会报错停下。
- `kamiak/waymo_submit.sh`：按顺序提交上面三步，前一步没出错才开始下一步。

要用户同意的下载：
- gcloud 安装包 `google-cloud-cli-linux-x86_64.tar.gz`：来源 dl.google.com，94,021,891 字节，2026-10-20 的 590.0.0 版。
- Python 包（PyPI，版本和办公室电脑一样）：numpy 2.3.5、pandas 2.3.3、pyarrow 22.0.0、protobuf 7.34.1、scipy 1.16.3、matplotlib 3.10.7、pillow 12.0.0。这 7 个约 129 MB；加上依赖和 conda 装的 Python 3.13，约 200 MB。
- Waymo 数据：验证集 202 个、训练集 798 个文件。按本地 3 段平均每段 1,008,350,027 字节推算，一共约 1.0 TB；准确大小下载作业开头会打出来。

要多少地方（按本地 3 段推算）：
- scratch：原始文件约 1.0 TB，提取结果（含全部图片）约 376 GB。
- 课题组目录：表格副本约 4.3 GB，SSM 结果约 1 GB。课题组默认 500 GB，模型已经占了 247.55 GiB，先看还剩多少。

要多久（估计）：
- 下载：dtn 上从 HuggingFace 下模型是每秒 53–113 MiB，按这个速度 1 TB 要 2.5–5.5 小时。
- 提取和算 SSM：办公室电脑上每段约 6 s 提取（不算写图片）、11 s 算 SSM。
- 实际（2026-10-21，验证集）：
  - 下载（作业 31350122）用了 9 分 41 秒，平均每秒 342.8 MiB，内存最多 356 MB。202 个文件、205,619,283,313 字节，大小全部和存储桶一致。
  - 21 块（作业 31350123）同时跑，每块 29 秒到 3 分 31 秒，内存最多 425 MB；合并（作业 31350124）4 秒。
  - 结果：202 段、39,987 帧（每段 181–199 帧）。

### 一次性准备
1. 看空间：
   ```bash
   df -H /data/lab/cejli
   du -sh --si /data/lab/cejli/sy
   lsworkspace
   ```
   - 2026-10-21 装东西之前：课题组目录配额 501 GB，已用 372 GB，剩 129 GB；其中 `sy` 占 369 GB（`models/` 266 GB、`conda_envs/` 45 GB、`hf_cache/` 41 GB）。scratch 还没有工作区，整个 scratch 剩 20 TB。
   - 装完 gcloud 和 Python 环境，剩 127 GB。
2. 装 gcloud（在计算节点上）：
   ```bash
   srun --partition=kamiak --account=cejli --time=00:20:00 --cpus-per-task=2 --mem=4G bash -c '
   set -euo pipefail               # 校验不通过就停，不解压、不安装
   cd /data/lab/cejli/sy
   curl -fO https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
   echo "351ed4a2812c5525c1cee00334257a42c020fcaffa27d3067c1eec0a618dd83f  google-cloud-cli-linux-x86_64.tar.gz" | sha256sum -c
   tar -xf google-cloud-cli-linux-x86_64.tar.gz
   ./google-cloud-sdk/install.sh --quiet --usage-reporting=false --command-completion=false --path-update=false
   ./google-cloud-sdk/bin/gcloud --version
   ./google-cloud-sdk/bin/gcloud info --format="value(basic.python_location)"'
   ```
   - 校验值是 Google 官网（docs.cloud.google.com/sdk/docs/downloads-versioned-archives）2026-10-21 列的 590.0.0 版的值。Google 出了新版，这个文件就会换，`sha256sum -c` 会报 FAILED：到那个页面查新的校验值再装。
   - 最后一行应该是 `.../google-cloud-sdk/platform/bundledpythonunix/bin/python3`，也就是用安装包自带的 Python。dtn1 上的系统 Python 是 3.9，gcloud 要 3.10 以上。
   - 装好以后，安装包（94 MB）可以删。
   - 2026-10-21 装好了（作业 31350055，cn170）：安装包 94,021,891 字节，校验通过；`gcloud --version` 是 Google Cloud SDK 590.0.0（core 2026.10.16、gsutil 5.37、bq 2.1.40、bundled-python3-unix 3.14.8）；用的是自带的 Python。装完占 532 MB。
   - 家目录里还有一套旧的 gcloud（`~/google-cloud-sdk`，562.0.0），`.bashrc` 把它加进了 PATH，所以直接打 `gcloud` 用的是旧的。脚本和这里的命令都写完整路径。两套共用 `~/.config/gcloud` 里的登录信息，里面已经有一个 2026-03-26 登录的 Google 账号。
3. 登录 Google。这一步你自己做，用在 waymo.com/open 注册过的那个 Google 账号：
   ```bash
   /data/lab/cejli/sy/google-cloud-sdk/bin/gcloud auth login --no-launch-browser
   /data/lab/cejli/sy/google-cloud-sdk/bin/gcloud storage ls "gs://waymo_open_dataset_v_1_4_3/individual_files/validation/*.tfrecord" | head -3
   ```
   - 要写 `*.tfrecord`：只写到 `validation/` 的话，存储桶里有一个和文件夹同名的空条目，前 3 行是它和一行标题，看不到文件名（2026-10-21）。
   - 第一条会给一个网址：在自己电脑的浏览器里打开，登录、同意，再把页面上的验证码贴回终端。
   - 登录信息存在家目录的 `~/.config/gcloud`，dtn 和计算节点都能用。
   - 第二条能列出 3 个文件名，就说明有权限。这两条都很短，在登录节点上跑就行。
4. 建 Python 环境（在计算节点上）：
   ```bash
   srun --partition=kamiak --account=cejli --time=00:30:00 --cpus-per-task=4 --mem=8G bash -c '
   set -eo pipefail                # 出错就停（不加 -u：conda 的脚本里有没赋值的变量）
   source /opt/apps/miniconda3/3.12/etc/profile.d/conda.sh
   conda create -p /data/lab/cejli/sy/conda_envs/waymo python=3.13 -y
   /data/lab/cejli/sy/conda_envs/waymo/bin/pip install numpy==2.3.5 pandas==2.3.3 pyarrow==22.0.0 protobuf==7.34.1 \
       scipy==1.16.3 matplotlib==3.10.7 pillow==12.0.0
   /data/lab/cejli/sy/conda_envs/waymo/bin/python -I -c "import sys, numpy, pandas, pyarrow, google.protobuf, scipy, matplotlib, PIL; print(sys.version.split()[0], numpy.__version__, pandas.__version__, pyarrow.__version__, google.protobuf.__version__, scipy.__version__, matplotlib.__version__, PIL.__version__)"'
   ```
   最后一行应该打印 `3.13.x 2.3.5 2.3.3 22.0.0 7.34.1 1.16.3 3.10.7 12.0.0`。
   - 2026-10-21 装好了（作业 31350056，cn170）：打印的是 `3.13.16 2.3.5 2.3.3 22.0.0 7.34.1 1.16.3 3.10.7 12.0.0`。装完占 812 MB。
   - pip 自动装的依赖：contourpy 1.4.0、cycler 0.12.1、fonttools 4.67.0、kiwisolver 1.5.1、packaging 26.3、pyparsing 3.3.3、python-dateutil 3.0.0、pytz 2026.5、six 1.17.0、tzdata 2026.5。

### 运行（在仓库根目录 `~/benchmark-claude-remote` 下）
```bash
git pull
WS=$(mkworkspace)               # 建 scratch，2 周后自动删
echo "$WS"; lsworkspace         # 记下路径和到期时间，写进本机的 CLAUDE.local.md
bash kamiak/waymo_submit.sh "$WS" validation
bash kamiak/waymo_submit.sh "$WS" training
squeue -u $USER
```
- 每个 split 三步：下载 → 提取、检查、算 SSM（每 10 段一个任务，最多同时 20 个）→ 合并。
- 结果在 `/data/lab/cejli/sy/waymo_v143/results/<split>/`：
  - `ssm/` 下面的文件和办公室电脑 `results/waymo_demo/` 一样：`frame_ssm.csv`、`scene_summary.csv`、`params.txt`、图；
  - `checks/` 是提取检查。
- 表格副本在 `/data/lab/cejli/sy/waymo_v143/tables/<split>/<段名>/`。scratch 删了以后还能用它重算 SSM，只是快照图里没有相机画面。
- scratch 上有：
  - `$WS/raw/<split>/`：原始文件、`bucket_listing.txt`（存储桶里每个文件的大小）、`DOWNLOAD_COMPLETE`；
  - `$WS/extracted/<split>/`：提取结果，含全部图片；
  - `$WS/ssm_parts/<split>/`：每块的结果。
- **2 周内要做完的事**：算完 SSM、看完数以后，决定留哪些图（只留每 0.5 s 一帧约 79 GB），复制到课题组目录。点云检查（`check_lidar_points.py`）要用原始文件，要做也得在这 2 周里做。
- 出问题：
  - 某一块失败：看 `logs/waymo-ssm_<作业号>_<块号>.out`。改好后只重交那一块：`sbatch --array=<块号> kamiak/waymo_ssm.sbatch "$WS" <split>`。再重交合并（命令就是 `waymo_submit.sh` 最后那条 `sbatch --wrap`）。还在等的合并作业会显示 `DependencyNeverSatisfied`，用 `scancel` 取消。
  - 下载中断（时限到了、网络断了）：再跑一次 `bash kamiak/waymo_submit.sh "$WS" <split>`。下好的文件、提取过的段都会跳过。
  - 下载作业因为内存不够被杀（日志里有 `oom`）：把 `waymo_download.sbatch` 里的 `CLOUDSDK_STORAGE_PROCESS_COUNT` 改成 1。
- 日志里可以不管的警告（2026-10-21 验证集）：
  - `check_v143_extract.py` 的 `RuntimeWarning: All-NaN slice encountered`：有一段的自车一帧都没对上地图车道，检查表里"离车道中心多远"那几格是空的。
  - `waymo_ssm_frames.py` 的 pandas `FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated`：同一段的 `ego_lane` 一列全空。pandas 2.3.3 拼表时不看全空的列，所以结果不受影响；以后换 pandas 版本要再看一下。
  - 这一段是 `5990032395956045002_6600_000_6620_000`：自车平均 1.25 m/s，199 帧都按设计退回"沿自车实际路线"找前车（`map_fallback`），每帧都找到了前车。

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
- InternVL 的日志里有 `The tokenizer you are loading ... with an incorrect regex pattern ... fix_mistral_regex=True`：误报，不用管。transformers 只要看到 `config.json` 里没有 `transformers_version` 就发这条（InternVL 的正好没有），不会改动分词器。2026-10-20 核对过：切词规则、词表、合并表都和 Qwen3-VL-32B 的一样，我们所有提示文字和小试回答共 75,233 token 切出来完全相同。
- 补答时 InternVL 报 500，服务日志里是 `Multi-modal cache miss ... P0/P1 cache drift`：前一个请求因为太长被拒，它的图片留在了服务前端的缓存里却没到后端，下一个用到同样图片的请求就出错。`kamiak/force_answer.sbatch` 里给 InternVL 加了 `--mm-processor-cache-gb 0`（关掉这个缓存，只影响速度）。
- 日志里说连不上 huggingface：模型已经在本地，在 sbatch 里 `conda activate` 那行后面加 `export HF_HUB_OFFLINE=1`。
