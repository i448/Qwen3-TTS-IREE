import torch
from abc import ABC, abstractmethod


class PatchHolder(ABC):
    """Base class for all IREE-specific patches."""

    def __init__(self):
        print("Initializing patches...")
        self._originals = {}

    @abstractmethod
    def apply(self):
        """Apply the patch logic."""
        pass

    @abstractmethod
    def revert(self):
        """Revert the patch to the original state."""
        pass

    def __enter__(self):
        self.apply()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.revert()


class TorchDiffPatch(PatchHolder):
    """
    `torch.diff` patch
    """

    def apply(self):
        if "torch.diff" not in self._originals:
            self._originals["torch.diff"] = torch.diff
            original_diff = torch.diff

            print(" - Applying TorchDiffPatch...")

            def patched_diff(input, n=1, dim=-1, prepend=None, append=None):
                if n != 1 or append is not None:
                    return original_diff(input, n, dim, prepend, append)
                if prepend is not None:
                    input = torch.cat([prepend, input], dim=dim)
                if dim == -1:
                    return input[..., 1:] - input[..., :-1]
                slices_after = [slice(None)] * input.ndim
                slices_after[dim] = slice(1, None)
                slices_before = [slice(None)] * input.ndim
                slices_before[dim] = slice(None, -1)
                return input[tuple(slices_after)] - input[tuple(slices_before)]

            torch.diff = patched_diff

    def revert(self):
        if "torch.diff" in self._originals:
            torch.diff = self._originals["torch.diff"]


class BooleanLogicPatch(PatchHolder):
    """
    Patch to avoid i1 (boolean) operations in SPIR-V/Vulkan.
    Converts boolean bitwise/logical ops into integer arithmetic.
    """

    def apply(self):
        if "torch.Tensor.__and__" in self._originals:
            return

        print("Applying boolean logic patch (Object-Oriented)...")

        # Save originals
        self._originals["torch.Tensor.__and__"] = torch.Tensor.__and__
        self._originals["torch.Tensor.__or__"] = torch.Tensor.__or__
        self._originals["torch.Tensor.__invert__"] = torch.Tensor.__invert__
        self._originals["torch.logical_and"] = torch.logical_and
        self._originals["torch.logical_or"] = torch.logical_or
        self._originals["torch.logical_not"] = torch.logical_not

        # Define Patches
        original_and = self._originals["torch.Tensor.__and__"]

        def patched_and(self_tensor, other):
            if self_tensor.dtype == torch.bool or (
                isinstance(other, torch.Tensor) and other.dtype == torch.bool
            ):
                self_int = (
                    self_tensor.to(torch.int8)
                    if self_tensor.dtype == torch.bool
                    else self_tensor
                )
                other_int = (
                    other.to(torch.int8)
                    if isinstance(other, torch.Tensor) and other.dtype == torch.bool
                    else other
                )
                res = self_int * other_int
                return res.to(torch.bool)
            return original_and(self_tensor, other)

        original_or = self._originals["torch.Tensor.__or__"]

        def patched_or(self_tensor, other):
            if torch.is_tensor(self_tensor) and self_tensor.dtype == torch.bool:
                other_t = other if torch.is_tensor(other) else torch.tensor(other)
                return ((self_tensor.to(torch.int32) + other_t.to(torch.int32)) > 0).to(
                    torch.bool
                )
            return original_or(self_tensor, other)

        original_invert = self._originals["torch.Tensor.__invert__"]

        def patched_invert(self_tensor):
            if torch.is_tensor(self_tensor) and self_tensor.dtype == torch.bool:
                return (1 - self_tensor.to(torch.int32)).to(torch.bool)
            return original_invert(self_tensor)

        # Apply to namespace
        torch.Tensor.__and__ = patched_and  # pyright: ignore
        torch.Tensor.__or__ = patched_or  # pyright: ignore
        torch.Tensor.__invert__ = patched_invert  # pyright: ignore

        def to_int(t):
            return (
                t.to(torch.int32)
                if (isinstance(t, torch.Tensor) and t.dtype == torch.bool)
                else t
            )

        def patched_logical_and(input, other, *, out=None):
            res = (to_int(input) * to_int(other)).to(torch.bool)
            if out is not None:
                out.copy_(res)
                return out
            return res

        def patched_logical_or(input, other, *, out=None):
            res = ((to_int(input) + to_int(other)) > 0).to(torch.bool)
            if out is not None:
                out.copy_(res)
                return out
            return res

        def patched_logical_not(input, *, out=None):
            res = (1 - to_int(input)).to(torch.bool)
            if out is not None:
                out.copy_(res)
                return out
            return res

        torch.logical_and = patched_logical_and
        torch.logical_or = patched_logical_or
        torch.logical_not = patched_logical_not

    def revert(self):
        if self._originals:
            print("Reverting boolean logic patch...")
            torch.Tensor.__and__ = self._originals["torch.Tensor.__and__"]
            torch.Tensor.__or__ = self._originals["torch.Tensor.__or__"]
            torch.Tensor.__invert__ = self._originals["torch.Tensor.__invert__"]
            torch.logical_and = self._originals["torch.logical_and"]
            torch.logical_or = self._originals["torch.logical_or"]
            torch.logical_not = self._originals["torch.logical_not"]


class RoPEInitPatch(PatchHolder):
    def apply(self):
        import transformers.modeling_rope_utils as rope_utils

        if not hasattr(rope_utils, "ROPE_INIT_FUNCTIONS"):
            return

        if "rope_init" not in self._originals:
            self._originals["rope_init"] = dict(rope_utils.ROPE_INIT_FUNCTIONS)

            def _default_rope_init(config, device):
                # Use the same pattern as existing RoPE classes
                dim = getattr(
                    config,
                    "head_dim",
                    getattr(config, "hidden_size", 1024)
                    // getattr(config, "num_attention_heads", 16),
                )

                # Extract rope_theta from config or sub-configs systematically
                base = getattr(config, "rope_theta", None)
                if base is None:
                    # Check common sub-config attributes
                    for sub_attr in [
                        "talker_config",
                        "decoder_config",
                        "encoder_config",
                    ]:
                        sub_config = getattr(config, sub_attr, None)
                        if sub_config:
                            base = getattr(sub_config, "rope_theta", None)
                            if base:
                                break

                # Fallback to model type-based defaults
                if base is None:
                    model_type = str(getattr(config, "model_type", "")).lower()
                    base = 1000000.0 if "talker" in model_type else 10000.0

                # Calculate inv_freq using the same pattern as existing classes
                inv_freq = 1.0 / (
                    base ** (torch.arange(0, dim, 2, device=device).float() / dim)
                )

                # Extract attention_factor with proper fallback
                mscale = getattr(config, "attention_factor", None)
                if mscale is None:
                    rope_scaling = getattr(config, "rope_scaling", None)
                    mscale = (
                        rope_scaling.get("attention_factor", 1.0)
                        if isinstance(rope_scaling, dict)
                        else 1.0
                    )

                return inv_freq, mscale

            rope_utils.ROPE_INIT_FUNCTIONS["default"] = _default_rope_init

    def revert(self):
        if "rope_init" in self._originals:
            import transformers.modeling_rope_utils as rope_utils

            rope_utils.ROPE_INIT_FUNCTIONS = self._originals["rope_init"]


class SDPAPatch(PatchHolder):
    """
    Patch for `torch.nn.functional.scaled_dot_product_attention`.
    Decomposes SDPA into primitive operations for better Vulkan compatibility.
    Handles GQA (repeating KV heads), sliding window, and various masks.
    """

    def apply(self):
        if "torch.nn.functional.scaled_dot_product_attention" in self._originals:
            return

        print("Applying SDPA patch...")
        self._originals["torch.nn.functional.scaled_dot_product_attention"] = (
            torch.nn.functional.scaled_dot_product_attention
        )
        original_sdpa = torch.nn.functional.scaled_dot_product_attention

        def patched_sdpa(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
            sliding_window=None,
            **kwargs,
        ):
            if dropout_p != 0.0:
                # Fallback to original if dropout is requested
                return original_sdpa(
                    query,
                    key,
                    value,
                    attn_mask,
                    dropout_p,
                    is_causal,
                    scale,
                    sliding_window=sliding_window,
                    **kwargs,
                )

            # 1. GQA Support: repeat k/v heads to match q heads
            if query.shape[1] != key.shape[1]:
                n_rep = query.shape[1] // key.shape[1]
                batch, num_kv_heads, slen, head_dim = key.shape
                # Use expand + reshape for zero-copy repeat (if possible)
                key = (
                    key[:, :, None, :, :]
                    .expand(batch, num_kv_heads, n_rep, slen, head_dim)
                    .reshape(batch, num_kv_heads * n_rep, slen, head_dim)
                )
                value = (
                    value[:, :, None, :, :]
                    .expand(batch, num_kv_heads, n_rep, slen, head_dim)
                    .reshape(batch, num_kv_heads * n_rep, slen, head_dim)
                )

            if scale is None:
                scale = query.size(-1) ** -0.5

            # 2. Main attention calculation
            attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale

            # 3. Causal masking
            if is_causal:
                L, S = query.size(-2), key.size(-2)
                mask = torch.ones(L, S, device=query.device, dtype=torch.bool).tril(
                    diagonal=0
                )
                if sliding_window is not None:
                    # Apply sliding window constraint
                    mask = mask & torch.ones(
                        L, S, device=query.device, dtype=torch.bool
                    ).triu(diagonal=1 - sliding_window)
                attn_weight = attn_weight.masked_fill(~mask, float("-inf"))

            # 4. Attention mask handling
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_weight = attn_weight.masked_fill(~attn_mask, float("-inf"))
                else:
                    # Handle float masks (convert binary 0/1 to additive 0/-inf if needed)
                    if attn_mask.max() > 0.0:
                        attn_mask = (1.0 - attn_mask.to(attn_weight.dtype)) * float(
                            "-inf"
                        )
                    attn_weight = attn_weight + attn_mask

            # 5. Output
            attn_weight = torch.softmax(attn_weight, dim=-1)
            return torch.matmul(attn_weight, value)

        torch.nn.functional.scaled_dot_product_attention = patched_sdpa

    def revert(self):
        if "torch.nn.functional.scaled_dot_product_attention" in self._originals:
            torch.nn.functional.scaled_dot_product_attention = self._originals[
                "torch.nn.functional.scaled_dot_product_attention"
            ]


class PreprocessMaskPatch(PatchHolder):
    """
    Patch for `transformers.masking_utils._preprocess_mask_arguments`.
    Handles cases where `past_key_values` is a list/tuple (legacy format)
    instead of a `Cache` object, providing fallbacks for mask size calculation.
    """

    def apply(self):
        import transformers.masking_utils as masking_utils

        target = "transformers.masking_utils._preprocess_mask_arguments"
        if target in self._originals:
            return

        print("Applying preprocess mask arguments patch...")
        self._originals[target] = masking_utils._preprocess_mask_arguments
        original_preprocess = masking_utils._preprocess_mask_arguments

        def patched_preprocess(
            config,
            input_embeds,
            attention_mask,
            cache_position,
            past_key_values,
            position_ids,
            layer_idx,
        ):
            if not hasattr(past_key_values, "get_mask_sizes") and isinstance(
                past_key_values, (list, tuple)
            ):
                # Fallback for old list/tuple format
                if len(past_key_values) > 0:
                    # Assumes [batch, num_heads, seq_len, head_dim]
                    past_length = past_key_values[0][0].shape[2]
                    kv_length = past_length + input_embeds.shape[1]
                    kv_offset = 0
                    return False, attention_mask, None, kv_length, kv_offset

            return original_preprocess(
                config,
                input_embeds,
                attention_mask,
                cache_position,
                past_key_values,
                position_ids,
                layer_idx,
            )

        masking_utils._preprocess_mask_arguments = patched_preprocess

    def revert(self):
        target = "transformers.masking_utils._preprocess_mask_arguments"
        if target in self._originals:
            import transformers.masking_utils as masking_utils

            masking_utils._preprocess_mask_arguments = self._originals[target]


class MaskingLogicPatch(PatchHolder):
    """
    Patch for `transformers.masking_utils.and_masks` and `or_masks`.
    Ensures that mask combining logic is captured correctly in JIT/tracing.
    """

    def apply(self):
        try:
            import transformers.masking_utils as masking_utils
        except ImportError:
            return

        target = "transformers.masking_utils.and_masks"
        if target in self._originals:
            return

        print("Applying masking utils patch...")
        self._originals["transformers.masking_utils.and_masks"] = (
            masking_utils.and_masks
        )
        self._originals["transformers.masking_utils.or_masks"] = masking_utils.or_masks
        self._originals[target] = True

        def patched_and_masks(*mask_functions):
            def and_mask(batch_idx, head_idx, q_idx, kv_idx):
                res = None
                for mask_fn in mask_functions:
                    m = mask_fn(batch_idx, head_idx, q_idx, kv_idx)
                    if res is None:
                        res = m
                    else:
                        res = res & m
                return res

            return and_mask

        def patched_or_masks(*mask_functions):
            def or_mask(batch_idx, head_idx, q_idx, kv_idx):
                res = None
                for mask_fn in mask_functions:
                    m = mask_fn(batch_idx, head_idx, q_idx, kv_idx)
                    if res is None:
                        res = m
                    else:
                        res = res | m
                return res

            return or_mask

        masking_utils.and_masks = patched_and_masks
        masking_utils.or_masks = patched_or_masks

    def revert(self):
        target = "transformers.masking_utils.and_masks"
        if target in self._originals:
            import transformers.masking_utils as masking_utils
            masking_utils.and_masks = self._originals[
                "transformers.masking_utils.and_masks"
            ]
            masking_utils.or_masks = self._originals[
                "transformers.masking_utils.or_masks"
            ]
            del self._originals[target]
            del self._originals["transformers.masking_utils.and_masks"]
            del self._originals["transformers.masking_utils.or_masks"]


class RoPEDynamoPatch(PatchHolder):
    """
    RoPE Patch (Dynamo-friendly) for Qwen3-TTS.
    Optimizes `rotate_half` and `apply_multimodal_rotary_pos_emb` for better
    compatibility with torch.compile / Dynamo.
    """

    def apply(self):
        try:
            import qwen_tts.core.models.modeling_qwen3_tts as modeling_pkg
        except ImportError:
            return

        target = "qwen_tts.modeling_qwen3_tts"
        if target in self._originals:
            return

        print("Applying RoPE patches (Dynamo-friendly)...")
        self._originals[f"{target}.rotate_half"] = modeling_pkg.rotate_half
        self._originals[f"{target}.apply_multimodal_rotary_pos_emb"] = (
            modeling_pkg.apply_multimodal_rotary_pos_emb
        )
        self._originals[f"{target}.apply_rotary_pos_emb"] = (
            modeling_pkg.apply_rotary_pos_emb
        )
        self._originals[target] = True

        def patched_rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        modeling_pkg.rotate_half = patched_rotate_half

        def patched_apply_multimodal_rotary_pos_emb(
            q, k, cos, sin, mrope_section=None, mrope_interleaved=False, unsqueeze_dim=1
        ):
            if cos.ndim >= 3 and cos.shape[0] == 3:
                if mrope_section is not None:
                    m_num = len(mrope_section)
                    if m_num == 3:
                        if mrope_interleaved:
                            new_cos = cos[0].clone()
                            new_sin = sin[0].clone()
                            for i in range(1, m_num):
                                n_pairs = mrope_section[i]
                                if n_pairs > 0:
                                    end = i + n_pairs * m_num
                                    new_cos[..., i:end:m_num] = cos[i, ..., i:end:m_num]
                                    new_sin[..., i:end:m_num] = sin[i, ..., i:end:m_num]
                            cos, sin = new_cos, new_sin
                        else:
                            mrope_section_channels = [s * 2 for s in mrope_section]
                            cos_split = cos.split(mrope_section_channels, dim=-1)
                            sin_split = sin.split(mrope_section_channels, dim=-1)
                            cos = torch.cat(
                                [s[i % 3] for i, s in enumerate(cos_split)], dim=-1
                            )
                            sin = torch.cat(
                                [s[i % 3] for i, s in enumerate(sin_split)], dim=-1
                            )
                    else:
                        cos, sin = cos[0], sin[0]
                else:
                    cos, sin = cos[0], sin[0]

            if cos.ndim == 3:
                cos = cos.unsqueeze(unsqueeze_dim)
                sin = sin.unsqueeze(unsqueeze_dim)

            q_embed = (q * cos) + (patched_rotate_half(q) * sin)
            k_embed = (k * cos) + (patched_rotate_half(k) * sin)
            return q_embed, k_embed

        modeling_pkg.apply_multimodal_rotary_pos_emb = (
            patched_apply_multimodal_rotary_pos_emb
        )
        modeling_pkg.apply_rotary_pos_emb = patched_apply_multimodal_rotary_pos_emb

    def revert(self):
        target = "qwen_tts.modeling_qwen3_tts"
        if target in self._originals:
            import qwen_tts.core.models.modeling_qwen3_tts as modeling_pkg
            modeling_pkg.rotate_half = self._originals[f"{target}.rotate_half"]
            modeling_pkg.apply_multimodal_rotary_pos_emb = self._originals[
                f"{target}.apply_multimodal_rotary_pos_emb"
            ]
            modeling_pkg.apply_rotary_pos_emb = self._originals[
                f"{target}.apply_rotary_pos_emb"
            ]
            del self._originals[target]


class RepeatKVPatch(PatchHolder):
    """
    Repeat KV Patch to fix UNPACK_SEQUENCE error for Qwen3-TTS.
    Provides a more robust implementation of `repeat_kv` that avoids sequence
    unpacking issues in certain JIT scenarios.
    """

    def apply(self):
        try:
            import qwen_tts.core.models.modeling_qwen3_tts as modeling_pkg
        except ImportError:
            return

        target = "qwen_tts.modeling_qwen3_tts.repeat_kv"
        if target in self._originals:
            return

        print("Applying repeat_kv patch...")
        self._originals[target] = modeling_pkg.repeat_kv

        def patched_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
            s = hidden_states.shape
            if len(s) == 4:
                batch, num_key_value_heads, slen, head_dim = s
            elif len(s) == 5:
                batch = s[0]
                num_key_value_heads = s[1]
                slen = s[-2]
                head_dim = s[-1]
                if n_rep == 1:
                    return hidden_states
                return torch.repeat_interleave(hidden_states, repeats=n_rep, dim=1)
            else:
                return hidden_states

            if n_rep == 1:
                return hidden_states

            hidden_states = hidden_states[:, :, None, :, :].expand(
                batch, num_key_value_heads, n_rep, slen, head_dim
            )
            return hidden_states.reshape(
                batch, num_key_value_heads * n_rep, slen, head_dim
            )

        modeling_pkg.repeat_kv = patched_repeat_kv

    def revert(self):
        target = "qwen_tts.modeling_qwen3_tts.repeat_kv"
        if target in self._originals:
            import qwen_tts.core.models.modeling_qwen3_tts as modeling_pkg
            modeling_pkg.repeat_kv = self._originals[target]
            del self._originals[target]


class CompositePatch(PatchHolder):
    """A collection of patches to be applied together."""

    def __init__(self, patches):
        self.patches = patches

    def apply(self):
        for patch in self.patches:
            patch.apply()

    def revert(self):
        for patch in reversed(self.patches):
            patch.revert()


class Patches:
    """Factory to manage and create IREE patches."""

    _REGISTRY = {
        "bool_logic": BooleanLogicPatch,
        "torch_diff": TorchDiffPatch,
        "rope_init": RoPEInitPatch,
        "sdpa": SDPAPatch,
        "mask_preproc": PreprocessMaskPatch,
        "mask_logic": MaskingLogicPatch,
        "rope_dynamo": RoPEDynamoPatch,
        "repeat_kv": RepeatKVPatch,
    }

    @classmethod
    def apply_patch(cls, names) -> PatchHolder:
        """
        Apply one or more patches.
        Args:
            names: A single patch name (str) or a list/tuple of patch names.
        Returns:
            A PatchHolder instance (or CompositePatch for multiple).
        """
        if isinstance(names, (list, tuple)):
            patches = [cls.apply_patch(name) for name in names]
            return CompositePatch(patches)

        patch_class = cls._REGISTRY.get(names)
        if not patch_class:
            raise ValueError(f"Unknown patch type: {names}")
        return patch_class()
