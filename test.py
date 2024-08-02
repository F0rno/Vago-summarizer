import tiktoken

def count_tokens_in_file(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    tokenizer = tiktoken.get_encoding("cl100k_base")  # Use the correct encoding name
    tokens = tokenizer.encode(text)
    num_tokens = len(tokens)
    return num_tokens

def tokens_of_n_characters(num_characters):
    text = "a" * num_characters
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


# Example usage
file_path = "transcriptions/math-transcript.txt"
num_tokens = count_tokens_in_file(file_path)
print(f"Number of tokens in file: {num_tokens}")
num_tokens = tokens_of_n_characters(100_000)
print(f"Number of tokens_of_n_characters in file: {num_tokens}")