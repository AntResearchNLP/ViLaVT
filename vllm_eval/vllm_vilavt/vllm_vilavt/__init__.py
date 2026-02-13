import os
from vllm import ModelRegistry
from transformers import AutoConfig
from .configuration_vilavt import VilavtConfig
from .qwen2_5_vl_vilavt import Qwen2_5_VLForConditionalGeneration_Vilavt

def register():
    # Test directly passing the model

    AutoConfig.register("vilavt", VilavtConfig, exist_ok=True)
   
    model_name="Qwen2_5_VLForConditionalGeneration_Vilavt"
    if "Qwen2_5_VLForConditionalGeneration_Vilavt" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "Qwen2_5_VLForConditionalGeneration_Vilavt",
            "vllm_vilavt.qwen2_5_vl_vilavt:Qwen2_5_VLForConditionalGeneration_Vilavt",
        )
        print(f"[PID {os.getpid()}] ✅ Successfully registered custom model '{model_name}' with vLLM.")

