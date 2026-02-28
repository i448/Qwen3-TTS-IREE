import torch
import iree

from export.talkerlm import TalkerLMWrapper
from export.tokenizerv2 import TokenizerV2Wrapper
from export.load import load_qwen3_tts


class Qwen3TTSDualTrackModels(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        shared_tts = load_qwen3_tts(model_name)
        self.talker_model = TalkerLMWrapper(shared_tts)
        self.tokenizer_model = TokenizerV2Wrapper(shared_tts)

    def encode(self, input_values, padding_mask):
        return self.tokenizer_model.encode(input_values, padding_mask)

    def decode(self, codes):
        return self.tokenizer_model.decode(codes)

    def speaker_forward(self, mels):
        """Expose speaker encoder: (B, T_mel, mel_dim) -> (B, D)"""
        return self.talker_model.speaker_encoder_forward(mels)

    def talker_forward(self, inputs_embeds, attention_mask):
        """Expose talker transformer: (B, S, D), (B, S) -> (B, S, D)"""
        return self.talker_model.talker_forward(inputs_embeds, attention_mask)


fxb = iree.turbine.aot.FxProgramsBuilder(Qwen3TTSDualTrackModels())  # pyright: ignore

batch = torch.export.Dim("batch", min=1, max=16)
seq_len = torch.export.Dim("seq_len", min=1, max=160000)
codes_len = torch.export.Dim("codes_len", min=1, max=10000)
mel_len = torch.export.Dim("mel_len", min=1, max=10000)

# Constants for 12Hz V2 and 0.6B Talker
num_quantizers = 16
mel_dim = 128
talker_dim = 1024


@fxb.export_program(
    args=(
        torch.empty((1, 1024), dtype=torch.float32),
        torch.empty((1, 1024), dtype=torch.int64),
    ),
    dynamic_shapes={
        "input_values": {0: batch, 1: seq_len},
        "padding_mask": {0: batch, 1: seq_len},
    },
)
def encode(module, input_values, padding_mask):
    return module.encode(input_values, padding_mask)


@fxb.export_program(
    args=(torch.empty((1, 128, num_quantizers), dtype=torch.long),),
    dynamic_shapes={
        "codes": {0: batch, 1: codes_len},
    },
)
def decode(module, codes):
    return module.decode(codes)


@fxb.export_program(
    args=(torch.empty((1, 256, mel_dim), dtype=torch.float32),),
    dynamic_shapes={
        "mels": {0: batch, 1: mel_len},
    },
)
def speaker_forward(module, mels):
    return module.speaker_forward(mels)


@fxb.export_program(
    args=(
        torch.empty((1, 128, talker_dim), dtype=torch.float32),
        torch.empty((1, 128), dtype=torch.int64),
    ),
    dynamic_shapes={
        "inputs_embeds": {0: batch, 1: seq_len},
        "attention_mask": {0: batch, 1: seq_len},
    },
)
def talker_forward(module, inputs_embeds, attention_mask):
    return module.talker_forward(inputs_embeds, attention_mask)
