import torch
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
)

class T5TextModel():
    def __init__(self, model_name="google/t5-v1_1-large", device="cuda", dtype=torch.bfloat16):
        self.max_length = 256
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.text_encoder, self.tokenizer = self.download_t5()

    def download_t5(self):
        text_encoder = T5EncoderModel.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
            use_safetensors=True,  
        )

        tokenizer = T5Tokenizer.from_pretrained(self.model_name)

        text_encoder.requires_grad_(False)
        text_encoder.eval()

        return text_encoder, tokenizer

    @torch.no_grad()
    def encoder(self, captions):
        if isinstance(captions, str):
            captions = [captions]

        tokens = self.tokenizer(
            captions,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        out = self.text_encoder(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask
        )

        text_emb = out.last_hidden_state.to(dtype=self.dtype)
        attn_mask = tokens.attention_mask.bool()

        return text_emb, attn_mask

    def to(self, device):
        self.text_encoder.to(device)
        self.device = str(device)
        return self    