"""Hooks that save what vLLM's Gemma 4 vision encoder receives (Gemma 4 31B-it check, 2026-11-12), put into each worker
with LLM.apply_model(install); rank 0 only, into $CAPTURE_DIR. In a module of its own so the workers can import it by
name (a function from the main script cannot be pickled across vLLM's processes; Qwen-Drive check, 2026-10-27).
- pe_NNNN.pt: the inputs of vision_tower.patch_embedder (pixel values as given, patch positions; -1 = padding);
- proj_NNNN.pt: the input of patch_embedder.input_proj (after 2 * (x - 0.5) and the cast to bfloat16);
- entry_NNNN.pt: the pixel values and positions at the model's _process_image_input (before the encoder; a second
  record in case the inner hooks do not run).
"""
import os


def install(model):
    import torch
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    d, n, m = os.environ['CAPTURE_DIR'], [0], [0]
    vt = getattr(model, 'vision_tower', None)
    pes = [vt.patch_embedder] if vt is not None else [x for x in model.modules()
                                                      if type(x).__name__ == 'Gemma4VisionPatchEmbedder']

    def cpu(x):
        return [t.detach().cpu() for t in x] if isinstance(x, (list, tuple)) else x.detach().cpu()

    def pre_pe(mod, args):
        if rank == 0:
            n[0] += 1
            torch.save(dict(pv=cpu(args[0]), pos=cpu(args[1])), os.path.join(d, f'pe_{n[0]:04d}.pt'))

    def pre_proj(mod, args):
        if rank == 0:
            torch.save(cpu(args[0]), os.path.join(d, f'proj_{n[0]:04d}.pt'))

    for p in pes:
        p.register_forward_pre_hook(pre_pe)
        p.input_proj.register_forward_pre_hook(pre_proj)

    orig = model._process_image_input

    def entry(image_input):
        if rank == 0:
            m[0] += 1
            torch.save(dict(pv=cpu(image_input['pixel_values']), pos=cpu(image_input['pixel_position_ids'])),
                       os.path.join(d, f'entry_{m[0]:04d}.pt'))
        return orig(image_input)

    model._process_image_input = entry
    return dict(rank=rank, patch_embedders=len(pes), model_class=type(model).__name__)
