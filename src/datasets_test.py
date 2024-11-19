import datasets
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AdamW,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
)


def main():
    data = datasets.load_dataset("roycehe/tieba", split="train")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print(tokenizer)
    print(data)
    print(data[0])

def tokenize_function(example):
    text = example["text"]
    text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")  # 去掉特殊标记
    return tokenizer(text, padding="max_length", truncation=True, max_length=1024)
main()
