#!/bin/bash
# Run the VLM pilot on Kamiak: start a vLLM server for an open-weight Qwen model on H100 GPUs, then send the requests
# with code/run_vlm.py. See kamiak/README.md for the one-time setup.
#
# 2026-10-19 verified against the live cluster (sinfo -p camas / scontrol show partition camas / scontrol show
# node sn16): camas = sn[16-18], 4x H100 80GB each, GRES type name is "tesla" (gpu:tesla:4), so a plain `gpu:2`
# request resolves fine since there's only one GRES type per node. MaxTime=7-00:00:00.
# All 3 nodes see real contention from other users - the job may queue.
#
# Submit from the repository root (Slurm does not create the log folder, so make it first):
#   mkdir -p logs && sbatch kamiak/qwen_vlm.sbatch                                  # MODEL=qwen3-vl-32b (default)
#   sbatch --export=ALL,MODEL=qwen3-vl-32b-thinking kamiak/qwen_vlm.sbatch          # other models: qwen3-vl-32b-thinking,
#                                                                                   # qwen3.6-27b, internvl3.5-38b
# Arguments after the script name are passed on to run_vlm.py and override the ones below, e.g. a small test run:
#   sbatch kamiak/qwen_vlm.sbatch --limit 2 --repeats 1 --out results/vlm_scores/qwen3-vl-32b/smoke_qwen3-vl-32b.jsonl
#SBATCH --job-name=qwen-vlm
#SBATCH --partition=camas              # 4x H100 80GB per node, GPU jobs not preemptable
#SBATCH --gres=gpu:2                   # 2 GPUs -> tensor parallel 2; confirmed syntax, see note above
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00              # within camas MaxTime=7-00:00:00; a timed-out run resumes when resubmitted
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

CONDA_ENV=/data/lab/cejli/sy/conda_envs/vllm_cu129           # vLLM 0.32.0 built for CUDA 12.9: Kamiak's driver
                                                              # (535, CUDA 12.2) cannot run the default cu130 build.
N_GPUS=2
MODEL=${MODEL:-qwen3-vl-32b}

# Per-model settings. Sampling = the model's own recommended values (its generation_config.json), sent explicitly so
# each output line records them. --max_tokens counts thinking tokens too.
case "$MODEL" in
    qwen3-vl-32b)   # Qwen/Qwen3-VL-32B-Instruct rev 0cfaf481 (2026-10-19, user's choice); dense, no thinking mode
        MODEL_DIR=/data/lab/cejli/sy/models/Qwen3-VL-32B-Instruct
        SERVE_ARGS=(--max-model-len 32768)
        SAMPLING=(--temperature 0.7 --top_p 0.8 --top_k 20 --max_tokens 1024)
        ;;
    qwen3-vl-32b-thinking)   # Qwen/Qwen3-VL-32B-Thinking rev 7edd10ff (2026-10-19, user's choice): same base and size
                             # as the Instruct model but thinks first, so the two isolate the effect of thinking.
                             # --reasoning-parser / --max_tokens / --max-model-len as for qwen3.6-27b below.
        MODEL_DIR=/data/lab/cejli/sy/models/Qwen3-VL-32B-Thinking
        SERVE_ARGS=(--max-model-len 40960 --reasoning-parser qwen3)
        SAMPLING=(--temperature 0.8 --top_p 0.95 --top_k 20 --max_tokens 24576)
        ;;
    qwen3.6-27b)    # Qwen/Qwen3.6-27B; thinks before answering (kept on, 2026-10-19). --reasoning-parser qwen3 keeps
                    # the thinking out of message.content (it lands in message.reasoning). Never use that parser for a
                    # model without thinking: it treats the whole reply as thinking and leaves the answer empty.
                    # --gdn-prefill-backend triton: its GDN layers would otherwise use a FlashInfer kernel that needs
                    # nvcc. The 2026-10-19 smoke test thought up to 10,896 tokens, so --max_tokens 24576; the longest
                    # prompt (sep, ~13,660 tokens) + 24576 fits in --max-model-len 40960.
        MODEL_DIR=/data/lab/cejli/sy/models/Qwen3.6-27B
        SERVE_ARGS=(--max-model-len 40960 --reasoning-parser qwen3 --gdn-prefill-backend triton)
        SAMPLING=(--temperature 1.0 --top_p 0.95 --top_k 20 --max_tokens 24576)
        ;;
    internvl3.5-38b)   # OpenGVLab/InternVL3_5-38B rev de99855b (2026-10-20, user's choice): InternViT-6B + Qwen3-32B,
                       # thinking off (its default). generation_config.json has no sampling values; 0.8 / 0.8 are the
                       # ones in the model card's API example (user's choice). --trust-remote-code: the config class
                       # ships as .py files with the weights (read 2026-10-20: configuration only). Images are tiled
                       # into up to 12 448x448 tiles + thumbnail (official default), so a sep prompt is ~35k tokens.
        MODEL_DIR=/data/lab/cejli/sy/models/InternVL3_5-38B
        SERVE_ARGS=(--max-model-len 40960 --trust-remote-code)
        SAMPLING=(--temperature 0.8 --top_p 0.8 --max_tokens 1024)
        ;;
    internvl3.5-38b-thinking)   # the same weights with thinking on (2026-10-20, user's choice to run both). Same
                                # messages as every other model (user's choice): instead of the R1 system prompt the
                                # model card uses to switch thinking on, kamiak/internvl_thinking_chat_template.jinja
                                # starts the reply with <think>, as Qwen's thinking templates do. (The reference smoke
                                # test with the R1 prompt, job 31327939, added --system_prefix_file
                                # code/internvl_thinking_system_prompt.txt.) Temperature 0.6 as the card recommends
                                # for thinking (it gives no top_p / top_k). The reply holds thinking + </think> +
                                # answer: the deepseek_r1 parser splits it. vLLM caps --max_tokens at the context
                                # left: ~5,900 tokens for a sep prompt.
        MODEL_DIR=/data/lab/cejli/sy/models/InternVL3_5-38B
        SERVE_ARGS=(--max-model-len 40960 --trust-remote-code --reasoning-parser deepseek_r1
                    --chat-template kamiak/internvl_thinking_chat_template.jinja)
        SAMPLING=(--temperature 0.6 --max_tokens 24576)
        ;;
    qwen-drive-1.0-4b)   # Qwen/Qwen-Drive-1.0-4B rev 28484089 (2026-10-26, user's choice): Qwen3.5-4B fine-tuned for
                         # driving. Only its vision-language model, from the folder kamiak/checks/make_vlm_folder.py
                         # made (vLLM does not know the QwenDriveForPlanning wrapper; the tensors are the original's).
                         # The same settings as qwen3.5-4b (user): thinking on (the chat template's default), the
                         # sampling of its generation_config.json, its own context of 32,768 (a sep prompt leaves
                         # ~19,000 tokens to think), --max_tokens as for the other thinking models; images capped by
                         # its preprocessor_config.json (vLLM's default). --gdn-prefill-backend triton as for
                         # qwen3.6-27b (the same Qwen3.5 architecture).
        MODEL_DIR=/data/lab/cejli/sy/models/Qwen-Drive-1.0-4B-vlm
        SERVE_ARGS=(--max-model-len 32768 --reasoning-parser qwen3 --gdn-prefill-backend triton)
        SAMPLING=(--temperature 0.6 --top_p 0.95 --top_k 20 --max_tokens 24576)
        ;;
    qwen3.5-4b)   # Qwen/Qwen3.5-4B rev 851bf6e8 (2026-10-26, user's choice): the model Qwen-Drive was fine-tuned from,
                  # as the control without driving training; exactly qwen-drive-1.0-4b's settings (user). Its repo
                  # has no generation_config.json; no presence_penalty (run_vlm.py does not send it).
        MODEL_DIR=/data/lab/cejli/sy/models/Qwen3.5-4B
        SERVE_ARGS=(--max-model-len 32768 --reasoning-parser qwen3 --gdn-prefill-backend triton)
        SAMPLING=(--temperature 0.6 --top_p 0.95 --top_k 20 --max_tokens 24576)
        ;;
    *) echo "unknown MODEL=$MODEL"; exit 1 ;;
esac
SERVED_NAME=$MODEL                                            # name recorded in every output line
OUT=results/vlm_scores/$MODEL/pilot_$MODEL.jsonl

source /opt/apps/miniconda3/3.12/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
# FlashInfer's top-p/top-k sampler would be JIT-compiled at first use, but nvcc is not where FlashInfer looks for it
# on Kamiak; this selects vLLM's own sampler (same math, possibly a little slower)
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p logs "results/vlm_scores/$MODEL"
PORT=$((8000 + SLURM_JOB_ID % 1000))
echo "MODEL=$MODEL MODEL_DIR=$MODEL_DIR OUT=$OUT"
# every run so far: camas (sn16-18 are identical), 2x H100 PCIe 80GB, driver 535.274.02; keep it that way (user,
# 2026-10-20) and record what this job got
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

# vLLM OpenAI-compatible server; up to 25 images per request, prompts of ~14k tokens
vllm serve "$MODEL_DIR" --served-model-name "$SERVED_NAME" --port "$PORT" \
    --tensor-parallel-size "$N_GPUS" --limit-mm-per-prompt '{"image": 25}' \
    --gpu-memory-utilization 0.90 "${SERVE_ARGS[@]}" \
    > "logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

echo "waiting for the vLLM server on port $PORT ..."
until curl -sf "http://localhost:$PORT/v1/models" > /dev/null; do
    sleep 15
    kill -0 $SERVER 2>/dev/null || { echo "vLLM server died, see logs/vllm_${SLURM_JOB_ID}.log"; exit 1; }
done
echo "server ready"

# all five pilot conditions, 5 repeats each (2,550 requests); already finished requests are skipped, so a rerun resumes.
# "$@" = extra arguments given to sbatch after the script name; argparse keeps the last value, so they override.
# 32 workers: the server queues whatever its KV cache cannot hold at once.
python -I code/run_vlm.py \
    --grid_targets results/vlm_inputs/targets.csv \
    --sep_targets results/vlm_inputs_separate/targets.csv \
    --out "$OUT" \
    --conditions grid,sep_time,sep_camera,ctrl_grid,ctrl_sep --repeats 5 \
    --base_url "http://localhost:$PORT/v1" --model "$SERVED_NAME" \
    "${SAMPLING[@]}" --top_logprobs 5 --workers 32 --send "$@"
