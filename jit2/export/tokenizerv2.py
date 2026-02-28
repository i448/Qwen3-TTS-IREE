import torch
from pathlib import Path
import sys

iree_impl_dir = str(Path(__file__).resolve().parent.parent.parent)
jit_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(iree_impl_dir)
sys.path.append(jit_dir)

try:
    from .patches import Patches
    from .load import TTSLoader
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)


class TokenizerV2Wrapper(TTSLoader):
    def __init__(self, model_or_path) -> None:
        super().__init__(model_or_path)

        self.model = self.tts.model.speech_tokenizer.model
        self.model._can_compile_fullgraph = True

        self.encoder = self.model.encoder
        self.decoder = self.model.decoder

    def encode(self, input_values: torch.Tensor, padding_mask: torch.Tensor):
        """Simplified encode method for export with proper masking"""

        Patches.apply_patch(["sdpa", "rope_init", "torch_diff", "bool_logic"])

        # input_values: [batch, seq_len]
        # padding_mask: [batch, seq_len]

        # Encoder expects [batch, 1, seq_len]
        encoded_frames = self.encoder.encode(
            input_values=input_values.unsqueeze(1), return_dict=True
        )
        audio_codes = encoded_frames.audio_codes[  # pyright: ignore
            :, : self.model.encoder_valid_num_quantizers
        ]

        # Traceable masking logic
        # Calculate actual lengths from padding mask
        actual_lengths = padding_mask.sum(dim=1) // self.model.encode_downsample_rate
        # Create indices for slicing
        max_len = audio_codes.shape[-1]
        indices = torch.arange(max_len, device=audio_codes.device).unsqueeze(0)
        mask = indices < actual_lengths.unsqueeze(1)
        # Apply mask and transpose
        audio_codes = audio_codes * mask.unsqueeze(1)
        return audio_codes.transpose(1, 2)  # Return [B, T, Q] format

    def decode(self, codes: torch.Tensor):
        """Simplified decode method for export"""

        Patches.apply_patch(["sdpa", "rope_init", "torch_diff", "bool_logic"])

        # codes: [batch, codes_length, num_quantizers]

        # Direct decoder call
        # Transpose to [B, Q, T] as internal decoder expects
        # We ensure it's not and-ing negative values if LongTensor
        audio_codes = torch.clamp(codes, min=0)
        audio_values = self.decoder.chunked_decode(audio_codes.transpose(1, 2)).squeeze(
            1
        )
        return audio_values
