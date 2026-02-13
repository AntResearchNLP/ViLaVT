from transformers import AutoConfig, Qwen2_5_VLConfig, Qwen3Config
class VilavtConfig(Qwen2_5_VLConfig):
    model_type = "vilavt"
    def __init__(
        self,
        integration_point = None,
        embedding_model_path="Qwen/Qwen3-Embedding-0.6B",
        text_encoder_config: Qwen3Config = None,
        **kwargs):

        self.integration_point=integration_point
        # 保存 text_encoder 的完整 config（用于重建）
        if text_encoder_config is None:
            text_encoder_config = AutoConfig.from_pretrained(embedding_model_path)
        self.text_encoder_config = text_encoder_config
        
        super().__init__(**kwargs)
    
    # def to_dict(self):
    #     """
    #     覆盖父类的 to_dict 方法，控制序列化行为
    #     """
    #     output = super().to_dict()
        
    #     # # 设置正确的 architectures
    #     # output['architectures'] = ["Qwen2_5_VLForConditionalGeneration_Vilavt"]
        
    #     # 移除自动生成的 text_config（如果不需要）
    #     if 'text_config' in output:
    #         del output['text_config']
        
    #     # 确保 text_encoder_config 被正确序列化
    #     if hasattr(self, 'text_encoder_config') and self.text_encoder_config is not None:
    #         if hasattr(self.text_encoder_config, 'to_dict'):
    #             output['text_encoder_config'] = self.text_encoder_config.to_dict()
    #         else:
    #             output['text_encoder_config'] = self.text_encoder_config
        
    #     return output