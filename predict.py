# Import model dan tokenizer LLaMa
from cog import BasePredictor, Input
from transformers import LlamaForCausalLM, AutoTokenizer
import torch

class Predictor(BasePredictor):
    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "huihui-ai/Llama-3.2-3B-Instruct-abliterated"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = LlamaForCausalLM.from_pretrained(self.model_name).to(self.device)

    def predict(self, prompt, max_length=50):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_length=max_length)
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output