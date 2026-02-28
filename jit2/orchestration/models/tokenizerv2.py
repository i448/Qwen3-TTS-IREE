import torch
import sys
from pathlib import Path

# Add paths for local imports
jit_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(jit_dir))

try:
    from export.tokenizerv2 import TokenizerV2Wrapper
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)


class TokenizerV2Pipeline:
    """
    Orchestration pipeline for the Qwen3-TTS Tokenizer V2.
    Separates high-level management and data handling from the traceable export modules.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device

        # The wrapper contains the core modules meant for export/inference
        self.wrapper = TokenizerV2Wrapper(model_path).to(device)
        self.wrapper.eval()

        # Metadata used for orchestration logic
        self.downsample_rate = getattr(
            self.wrapper.model,
            "encode_downsample_rate",
            getattr(self.wrapper.model.config, "encode_downsample_rate", 240),
        )
        self.num_quantizers = getattr(
            self.wrapper.model,
            "encoder_valid_num_quantizers",
            getattr(self.wrapper.model.config, "encoder_valid_num_quantizers", 4),
        )

    @torch.no_grad()
    def encode(self, audio_values: torch.Tensor):
        """
        Orchestrates the encoding process:
        1. Prepares inputs (batching, device movement).
        2. Calls the core encoder.
        3. Applies masking and quantizer selection.
        """
        if audio_values.ndim == 1:
            audio_values = audio_values.unsqueeze(0)  # [1, T]

        # Move to device
        audio_values = audio_values.to(self.device)

        # 2. Call core encoder
        # Note: We use the underlying model's encoder directly or through the wrapper
        # In a real IREE scenario, this would call the compiled VMFB
        encoded_frames = self.wrapper.encoder.encode(
            input_values=audio_values.unsqueeze(1),  # Encoder expects [B, 1, T]
            return_dict=True,
        )

        audio_codes = encoded_frames.audio_codes  # [B, Q, T_codes]  # pyright: ignore

        # 3. Apply orchestration-level logic (Slicing/Masking)
        # Select valid quantizers
        audio_codes = audio_codes[:, : self.num_quantizers, :]

        # Calculate actual code lengths from audio padding mask
        # Rounding down based on the downsample rate
        actual_lengths = padding_mask.sum(dim=1) // self.downsample_rate

        # Masking tokens that are outside the actual audio length
        batch_size, q, max_len = audio_codes.shape
        indices = torch.arange(max_len, device=self.device).unsqueeze(0)
        mask = indices < actual_lengths.unsqueeze(1)

        # Apply mask and transpose to common [B, T, Q] format
        audio_codes = audio_codes * mask.unsqueeze(1)
        return audio_codes.transpose(1, 2)

    @torch.no_grad()
    def decode(self, audio_codes: torch.Tensor):
        """
        Orchestrates the decoding process:
        1. Validates/Prepares codes.
        2. Calls the core decoder.
        3. Handles output trimming/clean-up.
        """
        # Ensure codes are in [B, Q, T] format for the decoder
        if audio_codes.shape[1] != self.num_quantizers:
            # Assuming input was [B, T, Q]
            audio_codes = audio_codes.transpose(1, 2)

        audio_codes = audio_codes.to(self.device).to(torch.long)
        # Clamp to ensure valid indices for the codebook
        audio_codes = torch.clamp(audio_codes, min=0)

        # 2. Call core decoder
        # chunked_decode is used for memory efficiency in some variants
        audio_values = self.wrapper.decoder.chunked_decode(audio_codes)

        # 3. Post-processing
        # Squeeze out the channel dim [B, 1, T] -> [B, T]
        if audio_values.ndim == 3 and audio_values.shape[1] == 1:
            audio_values = audio_values.squeeze(1)

        return audio_values
