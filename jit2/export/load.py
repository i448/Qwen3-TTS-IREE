from pathlib import Path
import torch
import sys

iree_impl_dir = str(Path(__file__).resolve().parent.parent.parent)
jit_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(iree_impl_dir)
sys.path.append(jit_dir)

try:
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
except ImportError as e:
    print(f"Import Failed: {e}")
    sys.exit(1)


def load_qwen3_tts(model_path: str) -> Qwen3TTSModel:
    return Qwen3TTSModel.from_pretrained(
        model_path, device_map="cpu", dtype=torch.float32
    )


class TTSLoader(torch.nn.Module):
    def __init__(self, model_or_path):
        super().__init__()
        if isinstance(model_or_path, str):
            self.tts = load_qwen3_tts(model_or_path)
        else:
            self.tts = model_or_path
