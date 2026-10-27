"""Pre-hooks that save what vLLM's vision encoder receives (Qwen-Drive check, 2026-10-27), put into each worker with
LLM.apply_model(install): the input of model.visual (float32 pixel values and grid_thw) and the input of
model.visual.patch_embed (after the cast to bfloat16), rank 0 only, into $CAPTURE_DIR. In a module of its own so the
workers can import it by name (a function defined in the main script cannot be pickled across vLLM's processes).
"""
import os


def install(model):
    import torch
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    d, n = os.environ['CAPTURE_DIR'], [0]

    def pre_visual(mod, args, kwargs):
        if rank == 0:
            n[0] += 1
            g = kwargs.get('grid_thw', args[1] if len(args) > 1 else None)
            torch.save(dict(x=args[0].detach().cpu(), grid_thw=g if isinstance(g, list) else g.detach().cpu().tolist()),
                       os.path.join(d, f'visual_{n[0]:03d}.pt'))

    def pre_patch(mod, args):
        if rank == 0:
            torch.save(args[0].detach().cpu(), os.path.join(d, f'patch_{n[0]:03d}.pt'))

    model.visual.register_forward_pre_hook(pre_visual, with_kwargs=True)
    model.visual.patch_embed.register_forward_pre_hook(pre_patch)
    return rank
