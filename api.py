import requests

class OllamaClient:
    def __init__(self, base_url, api_key, model="llama3.1"):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def call_llm(self, prompt, system_prompt, temperature=0.7):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "temperature": temperature,
            "eval_count": 2_000, # 2k tokens for the response
            "stream": False
        }
        response = requests.post(f"{self.base_url}/api/generate", json=data, headers=headers)
        response.raise_for_status()
        return response.json()["response"]