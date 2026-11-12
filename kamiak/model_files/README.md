# Model file lists

Official file lists (Hugging Face `api/models/<repo>/tree/<commit>?recursive=true`: path, size, git oid, and the
sha256 of LFS files) of every model the experiments used, saved so a deleted model folder can be downloaded again at
the same version and checked (`kamiak/checks/verify_hf_snapshot.py <folder> <list>`).

| file | repository | commit | local folder (`/data/lab/cejli/sy/models/`) |
|---|---|---|---|
| Qwen3-VL-32B-Instruct_0cfaf48….json | Qwen/Qwen3-VL-32B-Instruct | 0cfaf48183f594c314753d30a4c4974bc75f3ccb | deleted 2026-10-26 |
| Qwen3-VL-32B-Thinking_7edd10f….json | Qwen/Qwen3-VL-32B-Thinking | 7edd10ffd1196091948fb245ff63e406ccb2d4d1 | deleted 2026-10-26 |
| InternVL3_5-38B_de99855….json | OpenGVLab/InternVL3_5-38B | de99855be3642cd44fe97c9b72d70e5ce2c07f69 | InternVL3_5-38B |
| Qwen3.5-4B_851bf6e….json | Qwen/Qwen3.5-4B | 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a | Qwen3.5-4B |
| Qwen-Drive-1.0-4B_2848408….json | Qwen/Qwen-Drive-1.0-4B | 28484089a7cc8c335cf5089fb0745cf7c49b6eaa | Qwen-Drive-1.0-4B (root files only; assets/, perception/, planner-rl/, planner-sft/ were not downloaded) |
| Qwen3.6-27B_6a9e13b….json | Qwen/Qwen3.6-27B | 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 | Qwen3.6-27B (downloaded without --revision; the commit is the one in its .cache/huggingface/download metadata, the same for all 29 files) |
| gemma-4-31B-it_842da37….json | google/gemma-4-31B-it | 842da3794eaa0b77d5f08bae87a17459d91ff475 | gemma-4-31B-it (2026-11-12, `--revision` given; 11 of the 12 files: `.eval_results/mmmu_pro.yaml`, a 182-byte benchmark-score record not used to run the model, left out by design with `--exclude ".eval_results/*"`) |

The commits of the four kept folders are those in their `.cache/huggingface/download/*.metadata` (one commit per
folder). 2026-11-12 (srun 31699343): every local file matches its official checksum (36/36, 14/14, 14/14, 29/29).
gemma-4-31B-it, 2026-11-12 (download job 31699642 on dtn1, 7 min 41 s; srun 31699792): 11/11 local files match; the
check reports `.eval_results/mmmu_pro.yaml` as "not present locally", as intended. (`verify_hf_snapshot.py` labels a
line "sha256" only for `.safetensors` files, but compares every LFS file, `tokenizer.json` included, by its sha256.)

`Qwen-Drive-1.0-4B-vlm_local_sha256.txt`: sha256 and size of every file of `Qwen-Drive-1.0-4B-vlm`, a folder made
here, not downloaded: `kamiak/checks/make_vlm_folder.py` from Qwen-Drive-1.0-4B (config from its `vlm_config`, the
weights without the `vlm.` prefix), checked tensor by tensor with `kamiak/checks/check_vlm_folder.py` (2026-10-26). It
can be rebuilt from Qwen-Drive-1.0-4B with the same script.
