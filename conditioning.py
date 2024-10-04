from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel
from typing import List    


class CLIPTextEmbedder(nn.Module):
    """
    Constructor for the CLIPTextEmbedder class.
    """
    def __init__(self, version: str = "openai/clip-vit-large-patch14", device: str = "cuda:0", max_length: int = 77):
        """
        :param version: The name of the CLIP model to use. Defaults to "openai/clip-vit-large-patch14".
        :param device: The device to use for the model. Defaults to "cuda:0".
        :param max_length: The maximum length of the input text. Defaults to 77.
        """
        super().__init__()

        self.tokenizer = CLIPTokenizer.from_pretrained(version, clean_up_tokenization_spaces=True)
        self.text_model = CLIPTextModel.from_pretrained(version).to(device).eval()
        self.max_length = max_length
        self.device = device    
        
    def forward(self, prompts: List[str]):

        batch_encoding = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        
        return self.text_model(input_ids=tokens).last_hidden_state
