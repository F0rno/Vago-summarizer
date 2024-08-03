import tiktoken

def count_tokens_in_file(file_path, system_prompt):
    with open(file_path, "r") as file:
        text = file.read()
    tokenizer = tiktoken.get_encoding("cl100k_base")
    file_tokens = tokenizer.encode(text)
    system_prompt_tokens = tokenizer.encode(system_prompt)
    return (len(file_tokens), len(system_prompt_tokens))

def tokens_of_n_characters(num_characters):
    text = "F" * num_characters
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)