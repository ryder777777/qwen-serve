"""
EXL3 Ampere (sm_86) compatibility patch.

Root cause: CUDA conv1d_update_kernel in gdn.cu:1448 crashes on Ampere (sm_80/86/87).
Fix: Force Triton fallback for conv1d, disable BC GDN path, auto-alloc cache states.
"""
import os
import torch

_skip_patch = False
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    if major == 8 and minor in (0, 6, 7):
        _skip_patch = False
    else:
        _skip_patch = True

if not _skip_patch:
    import exllamav3.modules.gated_delta_net_fn.conv1d as _conv1d
    import exllamav3.modules.gated_delta_net as _gdn
    import exllamav3.cache.fp16 as _fp16

    # Force Triton conv1d
    def _triton_conv1d_update(mixed_qkv, conv_state, recurrent_slots, conv1d_weight,
                              conv1d_bias, history=False, params=None):
        bsz, dim, seqlen = mixed_qkv.shape
        if conv_state is None:
            assert not history
            conv_state = torch.zeros((bsz, dim, conv1d_weight.shape[-1]),
                                     dtype=torch.bfloat16, device=mixed_qkv.device)
            dummy_slots = True
        else:
            dummy_slots = False
        if _conv1d.has_triton:
            from exllamav3.util.tensor import buffered_arange
            if dummy_slots:
                recurrent_slots = buffered_arange(bsz, mixed_qkv.device)
            mixed_qkv = _conv1d.causal_conv1d_update_slotted_triton(
                mixed_qkv, conv_state, recurrent_slots,
                conv1d_weight, conv1d_bias,
                transpose_output=True, history=history,
            )
        else:
            raise RuntimeError("Ampere GDN conv1d crashes; Triton required")
        return mixed_qkv
    _conv1d.causal_conv1d_update = _triton_conv1d_update

    # Auto-alloc recurrent state
    _orig_gst = _gdn.GDNLayerState.get_state_tensors
    def _aa_gst(self):
        if self.device is None:
            dev = self.module.device if hasattr(self.module, 'device') else "cuda:0"
            self.alloc(dev)
        return _orig_gst(self)
    _gdn.GDNLayerState.get_state_tensors = _aa_gst

    # Skip BC path
    _orig_fwd = _gdn.GatedDeltaNet.forward
    def _no_bc_fwd(self, x, params):
        _bc, _sp = self.bc, self.bc_split
        self.bc = None; self.bc_split = False
        try: return _orig_fwd(self, x, params)
        finally: self.bc = _bc; self.bc_split = _sp
    _gdn.GatedDeltaNet.forward = _no_bc_fwd
    _gdn.causal_conv1d_update = _triton_conv1d_update

    # Auto-alloc KV cache
    _orig_gkv = _fp16.CacheLayer_fp16.get_kv
    def _aa_gkv(self, cs, bt, sw=-1):
        if self.k is None and self.shape is not None:
            self.alloc(bt.device if bt is not None else "cuda:0")
        return _orig_gkv(self, cs, bt, sw)
    _fp16.CacheLayer_fp16.get_kv = _aa_gkv

    _orig_uvd = _fp16.CacheLayer_fp16.update_kv_direct
    def _aa_uvd(self, cs, bt, k, v, l):
        if self.k is None and self.shape is not None:
            self.alloc(bt.device if bt is not None else "cuda:0")
        return _orig_uvd(self, cs, bt, k, v, l)
    _fp16.CacheLayer_fp16.update_kv_direct = _aa_uvd

print(f"exl3_ampere_patch: {'active (Ampere sm_{major}{minor})' if not _skip_patch else 'skipped'}")
