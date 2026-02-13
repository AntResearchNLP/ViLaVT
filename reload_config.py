import os
import sys
import json
import shutil
from pathlib import Path

from transformers import AutoProcessor
from safetensors.torch import load_file

from vllm_vilavt import VilavtConfig

# Add ViLaVT module path to Python search path
MODULE_PATH = "qwen_vl_finetune/vilavt"
if MODULE_PATH not in sys.path:
    sys.path.insert(0, MODULE_PATH)

from modeling_vilavt_v4_3_verl import (
    Qwen2_5_VLForConditionalGeneration_Vilavt,  # noqa: F401 (for config.architectures)
)


def load_sharded_safetensors(model_dir: str) -> dict:
    """
    Load a sharded safetensors model into a single state dict on CPU.

    Args:
        model_dir: Directory containing `model.safetensors.index.json`
                   and sharded `.safetensors` files.

    Returns:
        Merged state_dict loaded on CPU.
    """
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_file, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = sorted(set(index["weight_map"].values()))
    state_dict = {}

    for shard in shard_files:
        shard_path = os.path.join(model_dir, shard)
        # Load each shard on CPU and merge into a single dict
        state_dict.update(load_file(shard_path, device="cpu"))

    return state_dict


def convert(source_path: str, save_path: str) -> None:
    """
    Convert a ViLaVT fine-tuned checkpoint to be compatible with
    Qwen2_5_VLForConditionalGeneration_Vilavt, and copy weights/processor.

    Steps:
      1. Load VilavtConfig from `source_path`.
      2. Update `architectures` and attention implementation.
      3. Save the updated config to `save_path`.
      4. Copy processor files from `source_path` to `save_path`.
      5. Copy all model weight files and index files.
    """
    source = Path(source_path)
    target = Path(save_path)

    print("Loading and converting config...")
    config = VilavtConfig.from_pretrained(source)
    # Set the model class name used by Transformers
    config.architectures = ["Qwen2_5_VLForConditionalGeneration_Vilavt"]
    # Use flash attention v2 if available
    config._attn_implementation = "flash_attention_2"

    print(f"Saving updated config to {target}...")
    target.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(target)

    print("Copying processor files...")
    processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)
    processor.save_pretrained(target)

    print("Copying model weight files...")
    # Copy all .safetensors files
    for file in source.glob("*.safetensors"):
        shutil.copy2(file, target / file.name)

    # Copy all model*.bin files
    for file in source.glob("model*.bin"):
        shutil.copy2(file, target / file.name)

    # Copy index files (e.g., model.safetensors.index.json)
    for file in source.glob("*.index.json"):
        shutil.copy2(file, target / file.name)

    print(f"[ViLaVT] Config, processor, and weight files have been saved to: {target}")


if __name__ == "__main__":
    # TODO: replace these with your actual paths
    src = "path/to/vilavt_sft"
    dst = "path/to/vilavt_sft_modified"
    convert(src, dst)
