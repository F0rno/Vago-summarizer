from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def download_model(model_name, token=None):
	# Download and cache the model and tokenizer
	AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=token)
	AutoTokenizer.from_pretrained(model_name, use_auth_token=token)

if __name__ == "__main__":
	model_name = "google/gemma-7b-it"
	token = "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"  # Replace with your actual token
	download_model(model_name, token)