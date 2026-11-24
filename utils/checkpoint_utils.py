from tempfile import TemporaryDirectory
from time import sleep

import torch
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError
from transformers import AutoModelForCausalLM


def load_pythia_checkpoint(model_name, revision, device, scratch_dir=None):
    with TemporaryDirectory(prefix="pythia-checkpoint-", dir=scratch_dir) as directory:
        for attempt in range(3):
            try:
                info = HfApi().model_info(model_name, revision=revision)
                names = {item.rfilename for item in info.siblings}
                if "model.safetensors" in names:
                    weights = ["model.safetensors"]
                elif "model.safetensors.index.json" in names:
                    weights = ["model.safetensors.index.json", "model-*-of-*.safetensors"]
                elif "pytorch_model.bin" in names:
                    weights = ["pytorch_model.bin"]
                elif "pytorch_model.bin.index.json" in names:
                    weights = ["pytorch_model.bin.index.json", "pytorch_model-*-of-*.bin"]
                else:
                    raise ValueError(f"No supported weights at {model_name}@{revision}")
                snapshot_download(model_name, revision=info.sha, local_dir=directory,
                                  max_workers=1, allow_patterns=["config.json",
                                  "generation_config.json", *weights])
                break
            except (OSError, HfHubHTTPError, RuntimeError):
                if attempt == 2:
                    raise
                sleep(10 * (attempt + 1))
        model = AutoModelForCausalLM.from_pretrained(
            directory, torch_dtype=torch.float32, local_files_only=True,
        ).to(device).eval().requires_grad_(False)
        model.config.use_cache = False
    return model
