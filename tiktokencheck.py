import tiktoken
enc = tiktoken.get_encoding("gpt2")
print(enc.decode([529]))