import tiktoken

def count_tokens_in_file(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    num_tokens = len(tokens)
    return num_tokens

def tokens_of_n_characters(num_characters):
    text = "F" * num_characters
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)