import requests
from sys import exit
import logging

class OllamaClient:
    def __init__(self, base_url, api_key, model):
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
            "stream": False
        }
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=data, headers=headers)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.ConnectionError:
            logging.info("Connection error. Please check your network and try again.")
            exit(1)
        except requests.exceptions.HTTPError as http_err:
            return {"error": f"HTTP error occurred: {http_err}"}
        except Exception as err:
            return {"error": f"An error occurred: {err}"}