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
  - 上面两个（仓库 `Qwen/Qwen3-VL-32B-Instruct`、`Qwen/Qwen3-VL-32B-Thinking`）主实验跑完以后，用户 2026-10-26 决定删掉模型文件夹。官方文件清单（每个文件的大小和校验值，HF API 的 `tree/<版本>`）存在仓库 `kamiak/model_files/<模型>_<版本>.json`；10-26 删之前按版本重新取过，和 10-19 下载时存的逐字节一样，两个文件夹也各自 26/26 个文件核对一致。要补跑就按版本重下（第 3 步的命令加 `--revision`），再 `python -I kamiak/checks/verify_hf_snapshot.py <模型文件夹> kamiak/model_files/<模型>_<版本>.json` 逐个核对。
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
   - Qwen-Drive-1.0-4B（用户 2026-10-26 定加）：仓库 `Qwen/Qwen-Drive-1.0-4B`，`--revision 28484089a7cc8c335cf5089fb0745cf7c49b6eaa`，只下根目录（问答只用根目录的视觉语言模型）：加 `--exclude "assets/*" --exclude "perception/*" --exclude "planner-rl/*" --exclude "planner-sft/*"`（每个规则写一次 `--exclude`；先加 `--dry-run` 看会下哪些）。14 个文件 9,101,605,825 字节，2026-10-26 用了 100 秒。官方文件清单用不递归的 `tree/<版本>`，`verify_hf_snapshot.py` 只比文件夹最上一层。
     - vLLM 不认它的外壳（`QwenDriveForPlanning`），张量名前面多 `vlm.`：用 `kamiak/checks/make_vlm_folder.py` 另做一个只含视觉语言模型的文件夹 `/data/lab/cejli/sy/models/Qwen-Drive-1.0-4B-vlm/`（配置取原 `vlm_config`，权重去掉 `vlm.` 另存、数据部分逐字节照抄），再用 `kamiak/checks/check_vlm_folder.py <原文件夹> <新文件夹> --load` 逐个张量核对（`--load` 要用有 torch 和 safetensors 的环境，比如 `vllm_cu129`）。

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

## 正式实验的主实验（2026-10-22 用户定的；做法和理由见 CLAUDE.md "VLM 打分"最后）
脚本在 `kamiak/main_run/`（用户 2026-10-22 定的：仓库里只放脚本，数据和图在课题组目录，路径见下面"脚本用到的路径"）。都从仓库根目录、在计算节点上跑（`srun` 或 `sbatch`）。
1. 复制图（只做一次；2026-10-22 做过，作业 31364138）：对比实验清单 303 段每 0.5 s 一帧、5 个相机，从 scratch 的提取结果复制到课题组目录的表格副本里（已经有、大小一样的跳过）：
   `python -I kamiak/main_run/copy_images.py <scratch 工作区> vlm_frames/comparison_lead7_targets.csv /data/lab/cejli/sy/waymo_v143/tables`
2. 生成输入图（只做一次；2026-10-22 做过，作业 31366325，16 个 CPU 4 分 44 秒）：
   `python -I kamiak/main_run/make_main_inputs.py ~/benchmark-claude-remote /data/lab/cejli/sy/waymo_v143`
   - 写出两个 split 合起来的 `results_lead_final_excl/frame_ssm_validation_training.csv`（已经有就报错退出，不覆盖）、`vlm_inputs/segments/` 的链接、`vlm_inputs/{grid,separate}/` 的图和 `targets.csv`（主实验 241 段的全部目标帧）、`targets_main.csv`（主实验的 1,123 帧）、每张图的 sha256（`vlm_inputs/sha256_{grid,separate}.txt`）。
3. 提交。末尾的参数覆盖脚本里试跑的输入和输出，所以日志第一行打印的 `OUT=` 还是试跑的路径，实际以 `--out` 为准：
   ```bash
   V=/data/lab/cejli/sy/waymo_v143/vlm_inputs
   sbatch --export=ALL,MODEL=<MODEL> kamiak/qwen_vlm.sbatch --grid_targets $V/grid/targets_main.csv \
       --sep_targets $V/separate/targets_main.csv --out /data/lab/cejli/sy/waymo_v143/vlm_scores/<MODEL>/main_<MODEL>.jsonl
   ```
   - 用户定的顺序（2026-10-22）：先单独跑 Qwen3-VL-32B-Instruct，跑完分析走一遍；分析没出错、帧数是 1,084 和 1,123、5 种输入 × 9 个 SSM 都有结果（第 4 步的验收），剩下 4 个就一次排好，不等用户看数字。
   - 排法：每个加 `--dependency=afterany:<前一个的作业号>`（前一个不管成功还是出错，结束了就接着跑），`sbatch --parsable` 拿作业号；Qwen3.6-27B 放最后，加 `--time=5-00:00:00`（试跑乘 11 约 70 小时，超过脚本默认的 2 天）。思考的 3 个模型后面各插一个补答作业（第 4 步第 2 条），下一个模型接在补答后面（排好以后改依赖：`scontrol update JobId=<作业号> Dependency=afterany:<补答的作业号>`）。2026-10-22 排的作业号见 `docs/handoff.md`。
   - Qwen3.6-27B 开始跑以后（用户 2026-10-22 要求）：用开头几百条的平均大小推算整个文件，和课题组目录剩的空间比，可能不够就先告诉用户，不要等写满：`python -I kamiak/main_run/estimate_size.py /data/lab/cejli/sy/waymo_v143/vlm_scores/qwen3.6-27b/main_qwen3.6-27b.jsonl`（默认用前 300 条、推算 28,075 条，另给压缩版的估计和剩余空间）。在试跑文件上验证过，前 300、500 条推算的大小和实际差 −16% 到 +29%，所以按估计值加 30% 来判断；跑到约十分之一（`--lines 3000`）再估一次。
     - 2026-10-23 又在 Qwen3-VL-32B-Thinking 的主实验文件上验证（用户要求从 `.gz` 直接读，不另存解压文件：`python -I kamiak/main_run/estimate_size.py <(zcat <文件>.gz) --lines N`；读管道时"现在多大"和"剩多少空间"两项显示 0，不用看）：前 300 条估 10,795,390,110 字节，比实际 7,644,342,017 大 41.2%；前 3,000 条估 7,321,934,592，小 4.2%。压缩率估的是 24.7%、24.9%，实际 24.9%。
     - 偏差的原因：请求按清单里帧的顺序发（每帧 5 种输入 × 5 次），所以前 300 条差不多就是前 13 帧，估计主要看这几帧想得长不长（Thinking 在这 13 帧平均生成 1,756 token，全部 1,249；前 3,000 条来自 121 帧，1,197；`kamiak/checks/first_lines.py`）。5 个模型的顺序一样，Qwen3.6-27B 的前 300 条也差不多是这几帧。只用前 300 条，往两个方向都可能差很多。
   - 每个模型提交前，和它自己的试跑比问题文字、采样设置、vLLM 启动参数（`bash kamiak/checks/check_presubmit.sh`：几个版本的 `run_vlm.py` 不发送地组装请求、逐字节比；提交脚本现在会发的参数和试跑服务日志里的 `non-default args` 比；试跑文件每行记下的采样；模型文件夹和 vLLM 环境在试跑后有没有改过）。
   - 排上以后到跑完之前，不要改 `code/run_vlm.py`、`code/force_answer.py`、`kamiak/internvl_thinking_chat_template.jinja` 和课题组目录的输入图：作业开始时才读它们（提交脚本本身在提交时已经存进 Slurm，之后改了不影响）。
4. 跑完以后：
   1. 看日志最后一行 `done: N ok (M unparsable), K errors`；K 不是 0 就用同样的命令再交一次，只补发没做完的。
   2. 思考的 3 个模型还要补答：写到上限、没读出答案的回答，做法、收尾文字和设置都和试跑一样。用户 2026-10-22 定：只是让脚本读写主实验的文件就直接补，做法上有任何不同先问用户。
      - 程序和提交脚本都不用改：`kamiak/force_answer.sbatch` 末尾的参数传给 `code/force_answer.py`，覆盖试跑的文件名和路径。日志第一行打印的 `PILOT=`、`OUT=` 还是试跑的，实际以末尾的参数为准，`force_answer.py` 的第一行会打印它读的文件。
      - `force_answer.py` 只读不压缩的 jsonl，所以要排在压缩之前。
      ```bash
      S=/data/lab/cejli/sy/waymo_v143/vlm_scores
      sbatch --dependency=afterany:<这个模型主实验的作业号> --time=06:00:00 --export=ALL,MODEL=<MODEL> kamiak/force_answer.sbatch \
          --pilot $S/<MODEL>/main_<MODEL>.jsonl --grid_targets $V/grid/targets_main.csv \
          --sep_targets $V/separate/targets_main.csv --out $S/<MODEL>/main_<MODEL>_forced_answers.jsonl
      ```
      - 补答跑完、压缩之前核对（用户 2026-10-23 定用这个脚本）：`python -I kamiak/main_run/check_forced.py <main jsonl> <main forced jsonl> results/vlm_scores/<MODEL>/pilot_<MODEL>_forced_answers.jsonl`。看补的是不是正好是主实验里写到上限、没读出答案的那些条；接回去的思考和记下的原用量和主实验那条一样不一样；字段、采样、`max_tokens`、收尾文字和试跑补答比；finish_reason、解析、提示和回答的 token、相机写法。它只读不压缩的文件。
   3. 核对：`python -I kamiak/main_run/check_run.py <main jsonl> results/vlm_scores/<MODEL>/pilot_<MODEL>.jsonl vlm_frames/main_all9_targets.csv`（请求齐不齐、每行记的字段和设置、每组的 prompt token 和试跑一样不一样、解析）。相机写法：`python -I kamiak/checks/camera_variants.py <main jsonl[.gz]> vlm_frames/main_all9_targets.csv`（用户 2026-10-23 定：以后用仓库里的这份）。
   4. 压缩（补答跑完以后）：`bash kamiak/main_run/compress.sh <main jsonl>`：另存 `.gz`，核对解压出来逐字节一样，原文件挪到 `/data/lab/cejli/sy/waymo_v143/vlm_scores_to_delete/<MODEL>/`，由用户删。补答文件 `main_<MODEL>_forced_answers.jsonl` 也同样压缩。`.gz` 核对完设成只读：`chmod a-w <文件>.gz`（用户 2026-10-26 定）。压缩前先数每帧每种输入读出答案的次数：`python -I kamiak/checks/answer_counts.py vlm_frames/main_all9_targets.csv /data/lab/cejli/sy/waymo_v143/vlm_scores/<MODEL>`，有一次都没有的格就先停下报用户（用户 2026-10-26 定）。
   5. 分析：`bash kamiak/main_run/analyze.sh /data/lab/cejli/sy/waymo_v143/analysis/<名字>`：主结果 1,084 帧（`main_1084/`）和敏感性分析 1,123 帧（`sensitivity_1123/`），按片段重抽 2,000 次，分组用"全部""同一时段"（Waymo 的时段标签；天气分组还没定）和"同一段"。汇总表：`python -I kamiak/main_run/summarize.py <名字>/main_1084 <名字>/sensitivity_1123`。验收（帧数、每种分组 5 × 9 格的 τ-b、区间、成对一致率、SSM 之间差的区间都有值，没有 AUC）：`python -I kamiak/main_run/main_run_accept.py <名字>/main_1084 1084`，`sensitivity_1123` 那份给 1123。Spearman 只在"全部"那组算，分组的那几行空着是设计如此。每个文件夹另有模型两两之差的表 `model_differences.csv`（用户 2026-10-27 定；三对写在 `analyze_vlm_ssm.py` 的 `MODEL_PAIRS`，有模型不在这次分析里的那对跳过），核对：`python -I kamiak/checks/check_model_differences.py <名字>/main_1084`（`sensitivity_1123` 也一样）。
   6. 把 `logs/qwen-vlm_<任务号>.out` 和 `logs/vllm_<任务号>.log` 复制到 `/data/lab/cejli/sy/waymo_v143/vlm_scores/<MODEL>/logs/`；补答作业的 `logs/force-answer_<任务号>.out`、`logs/vllm_<任务号>.log` 也复制过去。

## 物理量实验（2026-10-28 用户定；设计见 CLAUDE.md "VLM 打分"）
问模型 5 个量（前车距离、自车速度、前车速度、自车加速度、前车加速度），每个量一条独立的请求。输入、系统提示、5 种输入方式、图和图注都和风险实验一样，只换最后的问题和 JSON（`code/run_vlm.py` 的 `QUANTITIES`）。采样、思考、上下文、max_tokens 照各模型风险实验的设置（提交脚本里那一段），只是不存候选概率。
- 提交（不用改提交脚本，末尾的参数覆盖）：`sbatch --export=ALL,MODEL=<MODEL> kamiak/qwen_vlm.sbatch --task quantities --top_logprobs 0 --out results/vlm_scores/<MODEL>/pilot_quantities_<MODEL>.jsonl`（试跑：102 帧 × 5 种输入 × 5 个量 × 5 次 = 12,750 条）。小试加 `--limit 1 --repeats 1`（25 条），输出叫 `smoke_quantities_<MODEL>.jsonl`。不加 `--task` 就还是风险实验。
- 每行多一列 `quantity`；`parsed` 是第一个含这个量的键、值是数字的 JSON（写成字符串的数字也转），负数和离谱的数照原样记；断点续跑时按（段、帧、输入、量、第几次）跳过已有的。
- 改程序以后核对风险实验的请求不变：`sbatch kamiak/checks/compare_risk_requests.sbatch`（改前的程序放在 `tmp_checks/run_vlm_compare/run_vlm_old.py`，路径写死在脚本里；`kamiak/checks/compare_risk_requests.py`：改前改后的 `run_vlm.py` 各用一个什么都不发的假接口跑一遍，每条要发的请求——提示文字、图片顺序和内容、采样参数——按发送顺序逐条比 sha256，输出行和不发送时的打印也比）。
- 真值：`python -I code/quantity_truth.py <frame_ssm.csv> <目标帧清单> /data/lab/cejli/sy/waymo_v143/tables <输出 csv> --ssm_params <params.txt>...`。距离 = `gap` + 这段的车头到前摄像头的距离（`footprint_from_lidars` 的车头 x 减 `camera_position` 的 FRONT x，从这段的 `context.json` 读，读不到就报错）；其余 4 个量是 `v_ego`、`v_lead`、`a_ego`、`a_lead`。
- 试跑报告：`python -I kamiak/checks/quantities_pilot_report.py <pilot_quantities jsonl> <真值 csv>`。
- 主实验（用户 2026-10-28 定）：上面的提交命令加 `--grid_targets $V/grid/targets_main.csv --sep_targets $V/separate/targets_main.csv --out /data/lab/cejli/sy/waymo_v143/vlm_scores/<MODEL>/main_quantities_<MODEL>.jsonl`（140,375 条）。跑完核对：`python -I kamiak/checks/check_quantities_run.py <文件> vlm_frames/main_all9_targets.csv`（每格 5 次齐不齐、设置、每个量 × 每种输入的读出率和写到上限的、没读出的按类数和举例、长度和用时），然后照风险实验第 4 步压缩、`.gz` 设只读、复制日志。主实验 1,123 帧的真值在 `/data/lab/cejli/sy/waymo_v143/quantities/truth_main.csv`（`code/quantity_truth.py`）。
- 分析：`bash kamiak/main_run/analyze_quantities.sh /data/lab/cejli/sy/waymo_v143/analysis/<名字> /data/lab/cejli/sy/waymo_v143/analysis/<风险分析的名字>`（`code/analyze_quantities.py`；主结果 1,084 帧、敏感性 1,123 帧；风险分取风险分析文件夹里的 `vlm_frame_scores.csv`，重抽的核对用它的 `agreement.csv`，所以风险分析里要有这几个模型）。验收：`python -I kamiak/checks/quantities_accept.py <名字>/main_1084 1084 /data/lab/cejli/sy/waymo_v143/quantities/truth_main.csv --intervals`（`sensitivity_1123` 那份给 1123）。汇总成报告用的几张表：`python -I kamiak/main_run/summarize_quantities.py <名字>/main_1084`（敏感性那份同样）。

## 脚本用到的路径（`kamiak/main_run/`、`kamiak/checks/`）
脚本里写死的、或者要当参数给的路径（用户 2026-10-22 要求写在这里）：
- `/data/lab/cejli/sy/conda_envs/waymo/bin/python`：处理 Waymo 数据和做分析用的 Python（`analyze.sh` 里写死）。vLLM 环境 `/data/lab/cejli/sy/conda_envs/vllm_cu129/` 只有 `kamiak/qwen_vlm.sbatch` 用。
- `/data/lab/cejli/sy/models/<模型>/`：下载的模型。`check_internvl_cut.py` 写死了 InternVL3_5-38B 的分词器；`verify_hf_snapshot.py` 的模型文件夹当参数给。
- `/data/lab/cejli/sy/waymo_v143/`（课题组目录，不会自动删）：
  - `tables/<split>/<段名>/`：每段的表格副本；对比实验那 303 段还有 `images/<相机>/`（每 0.5 s 一帧）。
  - `results*/<split>/ssm/`：各版 SSM 结果，现在用 `results_lead_final_excl/`；它下面的 `frame_ssm_validation_training.csv` 是两个 split 合起来的（`make_main_inputs.py` 生成，`analyze.sh` 写死）。`input_checks.py` 也读 `results_lead_final_excl/<split>/ssm/frame_ssm.csv`。
  - `vlm_inputs/`：正式实验的输入图（`make_main_inputs.py` 生成）。
  - `vlm_scores/<模型>/`：正式实验的回答 `main_<模型>.jsonl`，跑完压缩成 `.jsonl.gz`（`analyze.sh` 写死这个文件夹）。
  - `vlm_scores_to_delete/<模型>/`：压缩后挪开的原文件，等用户删（`compress.sh` 写死）。
  - `scene_labels_waymo.csv`：1,000 段 Waymo 自己的时段、地点、天气标签（`code/make_scene_labels.py` 生成，`weather` 列空着；`analyze.sh` 写死）。
  - `analysis/<名字>/`：正式实验的分析结果。
  - `to_office/`：传回办公室电脑用的下载文件夹（用户 2026-10-23 定）。只放符号链接（不复制、不解压），指向上面的结果、分析输入和分析结果；另有 `sha256.txt`，列出链接背后每个真文件的 sha256 和字节数（`bash kamiak/checks/to_office_sha256.sh`，在计算节点上跑，已有就不覆盖）。下载时要跟随链接：`scp -r` 默认跟随，`rsync` 要加 `-L`。
- `/data/lab/cejli/sy/tmp_checks/`：检查脚本的原件和它们的输出（表、图、日志）。仓库里的是原样复制的，以后改仓库里的。
- `/scratch/user/<netid>/<工作区>`：scratch 工作区（原始文件、全部提取结果，2 周后连数据自动删），当参数给；这个账号的路径和到期日写在 Kamiak 的 `CLAUDE.local.md`。
- `~/benchmark-claude-remote`：Kamiak 上的仓库。`kamiak/checks/` 里多数脚本用 `sys.path.insert(0, os.path.expanduser('~/benchmark-claude-remote/code'))` 导入 `code/` 里的程序（`wod_common.py` 等），所以在别的电脑上要先改这一行。`check_internvl_cut.py`、`check_parse_first.py`、`reparse_scores.py` 写死了 `/home/shuyi.wang/benchmark-claude-remote/` 下的 `code/` 和试跑结果 `results/vlm_scores/`；`input_checks.py` 读 `<仓库>/results/vlm_inputs*`（办公室电脑生成的试跑输入图）。
- 用 `python -I` 跑时，脚本所在的文件夹不在 `sys.path` 里；`acceptance2.py`、`followup_sheets.py` 自己把 `kamiak/checks/` 加进去，好导入 `front_sequence.py`、`lead_conditions.py`。

`kamiak/checks/`：算出过写进 `docs/handoff.md`、CLAUDE.md 的数的检查脚本（用户 2026-10-22 定的；handoff 里每个数旁边写着是哪个脚本）：

| 脚本 | 算了什么 |
|---|---|
| `split_checks.py` | 每个 split 合并后：段数、帧数、77 m 标注范围、自车一直对不上车道的段、整段为空的列 |
| `compare_demo_office.py` | 两台机器算的 `frame_ssm.csv` 比对，旧标准（2026-10-21，demo 3 段办公室电脑对 Kamiak） |
| `compare_frame_ssm.py` | 同上，新标准（2026-10-22 加了 1e-9 的绝对下限；先后用树状数组一遍数完） |
| `ego_lane_check.py` | 自车为什么对不上地图车道 |
| `reversing_check.py` | 自车、前车倒车的帧 |
| `small_ttc_check.py` | TTC 很小的帧：前车是不是路边停着、后来被自车开过去的车 |
| `lead_change_reasons.py` | 加横向重叠以后前车变了的帧和原因 |
| `lead_compare_counts.py` | 加横向重叠以后前车变化的帧数 |
| `overlap_details.py` | 还剩的 TTC < 3 s 的帧，两车横向重叠的细节 |
| `acceptance_snapshots.py` | 第一批验收图（A、B 两组） |
| `acceptance2.py` | 第二批验收图（R、S、T 三组） |
| `front_sequence.py` | 一辆车前后几秒的 FRONT 相机拼图（验收用；`acceptance2.py`、`followup_sheets.py` 导入它） |
| `lead_conditions.py` | 某辆车在某帧为什么是 / 不是前车（B18、A08） |
| `followup_sheets.py` | R02、R07、R10 的追查和拼图 |
| `trial_slow_rule.py` | 慢车规则的试算 |
| `slow_rule_breakdown.py` | 慢车规则在正式结果里去掉了哪些帧、为什么 |
| `merge_lidar_parts.py` | 点云检查分块结果的合并和汇总 |
| `ssm_value_frames.py` | 9 个、前车类 7 个指标都有数值的帧：帧数、段数、前车、TTC 范围、指标之间的 τ-b |
| `speed_noise.py` | 接近速度很小的帧、TTC 很大的帧、速度的测量误差 |
| `da_probe.py` | Waymo 域适应部分第一帧里有什么 |
| `input_checks.py` | 输入图：换字体后 demo 和试跑比、两种排字方式比、复制的图和 scratch 的图生成出来比 |
| `label_regions.py` | 换字体后网格图和试跑不同的像素，是不是都在相机名的角上 |
| `summarize_scores.py` | 试跑每个模型的回答概况（解析、思考长度、用时、同一帧 5 次的标准差） |
| `check_parse_first.py` | 解析改成取第一份 JSON 以后，以前的回答结果变没变 |
| `reparse_scores.py` | 按新解析办法重算一个结果文件的 `parsed`（先备份，有一条结果会变就不写） |
| `check_internvl_cut.py` | InternVL 思考版写到上限的回答，补答时上下文够不够 |
| `verify_hf_snapshot.py` | 下载的模型文件逐个和官方校验值比 |
| `inspect_weights.py` | 一个 safetensors 文件里的张量名（只读文件头）和参照模型的比：类型、个数、名字的前缀、去掉层号后哪些名字对不上 |
| `make_vlm_folder.py` | 从 Qwen-Drive-1.0-4B 做一个只含视觉语言模型、vLLM 能读的文件夹（用户 2026-10-26 定） |
| `install_env.sh` | Qwen-Drive 官方代码用的环境 `conda_envs/qwen_drive`（用户 2026-10-26 同意；版本照官方 `requirements.txt`，不装 flash-attn） |
| `gpu_env_test.py` | 上面那个环境在 H100 上能不能跑：torch、causal-conv1d、flash-linear-attention 的内核，官方代码装模型、生成几个 token |
| `diff_tokenizers.py` | 两个 `tokenizer.json` 差在哪：按 JSON 结构比，再比原文的开头和结尾 |
| `qwen_drive_compare/` | Qwen-Drive 的 vLLM 和官方代码贪心解码核对（`sbatch kamiak/checks/qwen_drive_compare/compare.sbatch` 里写的路径是 `tmp_checks/qwen_drive/compare/`）：组装请求、官方代码生成、vLLM 服务生成、vLLM 逐 token 打分并存视觉部分的输入、按用户的标准出报告 |
| `prep_compare_inputs.py` | Qwen-Drive 的 vLLM 和官方代码核对用的输入：一帧 5 种输入的 30 张图，按 `preprocessor_config.json` 缩到最终尺寸存成无损 PNG（只放课题组目录） |
| `precheck_pixels.py` | 同一份 PNG 经官方图片代码和 Hugging Face 处理器得到的格数、像素值（float32 和转成 bfloat16 以后）一不一样 |
| `compare_model_rows.py` | 两次分析里同一个模型的结果逐行比（加了模型以后原来那个模型的数变没变） |
| `compare_analysis_versions.py` | 同样的数据、分析程序改前和改后各跑一次：改前的每张表在改后的里逐行、逐字节比，`params.txt` 和日志列出不同的行，只在改后才有的文件列出来（用户 2026-10-27：加模型两两之差以后原来的表一张都不能变） |
| `check_quantities_run.py` | 物理量的运行文件：每格 5 次齐不齐、设置、每个量 × 每种输入的读出率和写到上限的、没读出的按类数（截断、null、文字、只有一个数、文字里给数等）并举例、回答和思考长度、用时（用户 2026-10-28 要的主实验核对） |
| `quantities_accept.py` | `analyze_quantities.py` 输出的验收：帧数、每格有值、差正好等于两者相减；另用 scipy / numpy 重算"全部"的 τ-b 和误差、用公式重算模型眼里的 TTC、THW、DRAC，重抽的核对（风险区间一字不差） |
| `compare_risk_requests.py` | 改 `run_vlm.py` 前后风险实验的每条请求逐条比（用不发送的假接口；用户 2026-10-28：加物理量任务以后风险的请求要逐字节一样） |
| `quantities_pilot_report.py` | 物理量试跑的报告：每个量 × 每种输入的读出率、长度、用时、写到上限的；回答的分布；同一帧几次的离散程度；真值；每帧中位数和真值的 τ-b、误差，加速度三类分对的比例（只作参考） |
| `check_model_differences.py` | `model_differences.csv`：每对的行数，两个模型的值和 `agreement.csv` 里各自的 τ-b 是不是一字不差，差是不是正好等于两者相减，缺值、重抽次数，再按每对、每种分组打出"差 [区间]"的表 |
| `check_vlm_folder.py` | 上面那个文件夹和原文件夹逐个张量核对：名字、类型、形状、位置，每个张量的字节，用 safetensors 读出来 `torch.equal`，配置和复制的文件 |
| `camera_variants.py` | 一个运行文件里 `main_hazard_camera` 没按规定写法的回答：每种输入、每种写法几次，几次落在主结果拿掉的 39 帧上。没读出答案的、读出答案没写相机的另外各列一行，合计只算写法不同的（用户 2026-10-23 定；原来遇到没读出答案的回答会报错退出） |
| `count_cut_replies.py` | 一个运行文件里因长度停下的回答：每种输入里写到 `max_tokens` 的、撑满上下文的各几条，其中没读出答案的、陷进重复的（最后 1,000 个 token 里至少 95% 的步第一候选概率 ≥ 0.99）各几条；同一规则在正常结束的回答上误判几条（用户 2026-10-26 要两个 4B 模型的试跑报告单列这几项） |
| `answer_counts.py` | 每帧每种输入 5 次里读出答案的次数（照分析程序 `read_model` 读，补答并进去）：各有几格、最少几次、不满 5 次的格在哪种输入、有没有一次都没有的格 |
| `unparsable_report.py` | 一个运行文件里没读出答案的每一条：条件、repeat、finish_reason、生成 token 数、回答原文；是写到 max_tokens、撑满上下文，还是写完了但解析规则读不出（照 `run_vlm.parse` 逐个试回答里的 `{`，写出每处为什么不算答案） |
| `to_office_sha256.sh` | `to_office/` 里链接背后每个真文件的 sha256 和字节数，写进 `to_office/sha256.txt` |
| `first_lines.py` | 一个运行文件的前 N 行来自几帧、平均生成多少 token、每行多大，和整个文件比（`estimate_size.py` 为什么估偏） |
| `check_presubmit.sh` | 正式实验剩下 4 个模型提交前和各自试跑的比对（问题文字、vLLM 参数、采样、模型和环境有没有改过）；写死了 `tmp_checks/presubmit/`，`check_pilot_settings.py` 也从那里调 |
| `check_pilot_settings.py` | 试跑文件每行记下的采样设置和每个 token 的候选概率个数（`check_presubmit.sh` 调它） |

没放进仓库的（还在 `tmp_checks/`）：只用来调试环境、没有记下数的 `gpu_sanity_check.sh`、`gpu_sanity.sbatch`、`check_prefill_template.py`、`check_internvl_prefill.py`、`check_internvl_tokenizer.py`、`compare_tokenizers.py`；分析程序的回归测试 `analyze_regression/`、`analyze_regression2/`；`compress.sh`、`analyze.sh` 修正后的测试 `script_fix_tests/`；`camera_variants.py` 2026-10-23 改后在造的 7 行上的测试 `main_checks/camera_test/`；`unparsable_report.py` 在造的 6 条回答上的测试 `main_checks/unparsable_test/`；模型两两之差那张表的测试和正式比对 `model_diff/`（`analyze_vlm_ssm_without.py` 是加表以前 37cfa3b 的分析程序，`analyze_without.sh` 是只把程序换成它的 `analyze.sh`，`test.sbatch` 是 20 次重抽的测试）。

## Waymo 训练集 + 验证集：下载、提取、算 SSM（2026-10-21 写好，当天两部分都跑完）
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
- 实际（2026-10-21）：
  - 验证集：下载（作业 31350122）9 分 41 秒，平均每秒 342.8 MiB，内存最多 348 MiB；202 个文件、205,619,283,313 字节，大小全部和存储桶一致。21 块（31350123）同时跑，每块 29 秒到 3 分 31 秒，内存最多 415 MiB；合并（31350124）4 秒。结果 202 段、39,987 帧（每段 181–199 帧）。
  - 训练集：下载（31350305）41 分 12 秒，平均每秒 316.6 MiB；798 个文件、816,376,301,297 字节，大小全部一致。80 块（31350306，最多同时 20 块）一共约 10 分钟，每块 59 秒到 3 分 15 秒，内存最多 425 MiB；合并（31350307）12 秒。结果 798 段、158,081 帧（每段 171–200 帧）。
  - 占的地方：scratch 上原始文件 206 GB + 817 GB，提取结果 75 GB + 292 GB；课题组目录里表格副本 2.5 GB、结果 1.3 GB。
  - 训练集 `params.txt` 里有 1 段"最远的框近端"是 `nan`：那段一个车、行人、骑车人的 3D 框都没有（只有交通标志），没有框可量，不是出错（`docs/handoff.md` 第 2 节最后）。

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
- 每个 split 三步：下载 → 提取、检查、算 SSM（每 10 段一个任务，最多同时 20 个）→ 合并。已经下载完（有 `DOWNLOAD_COMPLETE`）的 split 不再交下载作业。
- 改了程序要重算、又要留着旧结果时：`RUN=<名字> bash kamiak/waymo_submit.sh "$WS" <split>`，结果写到 `$WS/ssm_parts_<名字>/` 和 `/data/lab/cejli/sy/waymo_v143/results_<名字>/`；提取过的段跳过，只重新检查和算 SSM。2026-10-21 用 `RUN=lead_overlap` 重算了前车加横向重叠、标出倒车以后的结果。
- 结果在 `/data/lab/cejli/sy/waymo_v143/results/<split>/`：
  - `ssm/` 下面的文件和办公室电脑 `results/waymo_demo/` 一样：`frame_ssm.csv`、`scene_summary.csv`、`params.txt`、图；
  - `checks/` 是提取检查。
- 表格副本在 `/data/lab/cejli/sy/waymo_v143/tables/<split>/<段名>/`。scratch 删了以后还能用它重算 SSM，只是快照图里没有相机画面。
- scratch 上有：
  - `$WS/raw/<split>/`：原始文件、`bucket_listing.txt`（存储桶里每个文件的大小）、`DOWNLOAD_COMPLETE`；
  - `$WS/extracted/<split>/`：提取结果，含全部图片；
  - `$WS/ssm_parts/<split>/`：每块的结果。
- **2 周内要做完的事**：算完 SSM、看完数以后，决定留哪些图（只留每 0.5 s 一帧约 79 GB），复制到课题组目录。点云检查（`check_lidar_points.py`）要用原始文件，要做也得在这 2 周里做。
- 点云检查（2026-10-22 在新规则的结果上做完）：每块一个任务，用那一块的 `frame_ssm.csv`，结果写在块旁边的 `lidar_check/`：
  ```bash
  sbatch --array=0-20%20 --export=ALL,RUN=lead_overlap kamiak/waymo_lidar_check.sbatch "$WS" validation
  sbatch --array=0-79%20 --export=ALL,RUN=lead_overlap kamiak/waymo_lidar_check.sbatch "$WS" training
  ```
  - 每段约 20 s，每块最多 3 分 11 秒，内存 8G 足够。合并和汇总用 `kamiak/checks/merge_lidar_parts.py`（结果见 `docs/handoff.md` 第 2 节最后）。
- 出问题：
  - 某一块失败：看 `logs/waymo-ssm_<作业号>_<块号>.out`。改好后只重交那一块：`sbatch --array=<块号> kamiak/waymo_ssm.sbatch "$WS" <split>`。再重交合并（命令就是 `waymo_submit.sh` 最后那条 `sbatch --wrap`）。还在等的合并作业会显示 `DependencyNeverSatisfied`，用 `scancel` 取消。
  - 下载中断（时限到了、网络断了）：再跑一次 `bash kamiak/waymo_submit.sh "$WS" <split>`。下好的文件、提取过的段都会跳过。
  - 下载作业因为内存不够被杀（日志里有 `oom`）：把 `waymo_download.sbatch` 里的 `CLOUDSDK_STORAGE_PROCESS_COUNT` 改成 1。
- 日志里可以不管的警告（2026-10-21 验证集第 15 块，都来自 `5990032395956045002_6600_000_6620_000`：这段自车走的路地图里没有车道，199 帧一帧都没对上，见 `docs/handoff.md` 第 2 节最后）：
  - `check_v143_extract.py:83: RuntimeWarning: All-NaN slice encountered`（`ego_to_lane_with_offset_median_m=np.nanmedian(d_with)`），第 85 行同样一条（`ego_to_lane_without_offset_median_m=np.nanmedian(d_without)`）。
    - 原因：检查程序找 10 m 内、方向差 45° 内的车道中心线，这段一帧都没有，`np.nanmedian` 对全空的数组返回 NaN 并发这条警告。第 84 行的 90% 分位数先判断了有没有值，所以没有警告。
    - 影响：只是检查表 `checks/extract_checks.csv` 里这段的这 3 格是空的，本来就该是空的。检查表不进 SSM 的计算。
  - `waymo_ssm_frames.py:424: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.`（`fr = pd.concat(frames, ignore_index=True)`）
    - 原因：这段的 `ego_lane` 一列全空（只有这一列）。
    - 影响：pandas 2.3.3 定拼接后的列类型时不看全空的列，结果是数值列；以后的 pandas 会把它算进去，`ego_lane` 可能变成 object 类型，csv 里车道编号的写法可能从 `123.0` 变成 `123`，数值不变。别的列不受影响。
  - 顺带：合并后的 `frame_ssm.csv` 里 `ego_lane` 本来就有两种写法（验证集 29,143 格写成 `123.0`，10,295 格写成 `123`），看那一块里有没有空值。读进来都是同样的数。

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
