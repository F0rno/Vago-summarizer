from openai import OpenAI
from enum import Enum

class Model(Enum):
    LLAMA_3_8B = "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"
    LLAMA_3_1_8B = "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"

class OpenAIClient:
    def __init__(self, base_url, api_key):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def call_llm(self, prompt, system_prompt, model=Model.LLAMA_3_1_8B, temperature=0.7):
        completion = self.client.chat.completions.create(
            model=model.value,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
        )
        return completion.choices[0].message.content
