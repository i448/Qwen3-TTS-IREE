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


class TalkerLMWrapper(TTSLoader):
    def __init__(self, model_or_path) -> None:
        super().__init__(model_or_path)
        self.model = self.tts.model

        self.talker = self.model.talker
        self.speaker_encoder = self.model.speaker_encoder  # For Base Variant models
        self.speech_tokenizer = self.model.speech_tokenizer
        self.padding_idx = getattr(
            self.model.config,
            "tts_pad_token_id",
            getattr(self.model.config, "pad_token_id", None),
        )

    def speech_tokenizer_encode(
        self, input_values: torch.Tensor, padding_mask: torch.Tensor
    ):
        """Simplified encode method for export"""

        Patches.apply_patch(
            [
                "bool_logic",
                "torch_diff",
                "rope_init",
                "sdpa",
                "mask_preproc",
                "mask_logic",
                "rope_dynamo",
                "repeat_kv",
            ]
        )

        # input_values: [batch, seq_len]
        # padding_mask: [batch, seq_len]

        # Access the underlying model inside the Qwen3TTSTokenizer wrapper
        tokenizer_model = self.speech_tokenizer.model

        # Encoder expects [batch, 1, seq_len]
        encoded_frames = tokenizer_model.encoder.encode(
            input_values=input_values.unsqueeze(1), return_dict=True
        )
        audio_codes = encoded_frames.audio_codes[
            :, : tokenizer_model.encoder_valid_num_quantizers
        ]

        # Traceable masking logic
        actual_lengths = (
            padding_mask.sum(dim=1) // tokenizer_model.encode_downsample_rate
        )
        max_len = audio_codes.shape[-1]
        indices = torch.arange(max_len, device=audio_codes.device).unsqueeze(0)
        mask = indices < actual_lengths.unsqueeze(1)
        audio_codes = audio_codes * mask.unsqueeze(1)

        return audio_codes.transpose(1, 2)  # [B, T, Q]

    def speaker_encoder_forward(self, mels: torch.Tensor):
        """Expose speaker encoder for export"""

        Patches.apply_patch(
            [
                "bool_logic",
                "torch_diff",
                "rope_init",
                "sdpa",
                "mask_preproc",
                "mask_logic",
                "rope_dynamo",
                "repeat_kv",
            ]
        )

        # mels: (B, mel_dim, T_mel)
        # Internal speaker_encoder expects (B, T_mel, mel_dim) because it transposes it to (B, mel_dim, T_mel) internally
        return self.speaker_encoder(mels.transpose(1, 2))

    def talker_forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        """Expose talker (causal transformer) for export"""

        Patches.apply_patch(
            [
                "bool_logic",
                "torch_diff",
                "rope_init",
                "sdpa",
                "mask_preproc",
                "mask_logic",
                "rope_dynamo",
                "repeat_kv",
            ]
        )

        # inputs_embeds: (B, seq_len, D)
        # attention_mask: (B, seq_len)
        return self.talker(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
