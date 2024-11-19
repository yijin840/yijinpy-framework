import datasets
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AdamW,
    DataCollatorForLanguageModeling,
)
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

# 加载 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# 分词函数
def tokenize_function(example):
    print(f"原始数据: {example['text'][:50]}")  # 打印原始文本的前 50 个字符
    cleaned_texts = [
        text.replace("<|im_start|>", "").replace("<|im_end|>", "") for text in example["text"]
    ]
    print(f"清理后的数据: {cleaned_texts[:1]}")  # 打印清理后的第 1 条数据
    tokens = tokenizer(cleaned_texts, padding="max_length", truncation=True, max_length=256)
    print(f"分词结果 (input_ids 前 10 个): {tokens['input_ids'][0][:10]}")  # 打印分词结果
    return tokens

# 主函数
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    # 加载数据集
    dataset = datasets.load_dataset("roycehe/tieba", split="train")
    print(f"加载数据集成功，共有 {len(dataset)} 条数据")
    dataset = dataset.select(range(100))  # 截断为前 1000 条
    print(f"截断后的数据集条数: {len(dataset)}")

    # 数据分词与格式化
    data = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    data.set_format(type="torch", columns=["input_ids", "attention_mask"])
    print(f"数据格式化完成，第 1 条数据: {data[0]}")

    # 初始化模型
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.pad_token_id).to(device)
    print("模型加载成功")

    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # 初始化数据加载器
    dataloader = DataLoader(
        data,
        collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        batch_size=4
    )
    print("数据加载器初始化成功")

    # 训练
    print("开始训练...")
    progress_bar = tqdm(total=len(dataloader), desc="训练进度")
    for i, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        max_index = input_ids.max().item()
        if max_index >= tokenizer.vocab_size:
            print(f"检测到索引越界，最大索引: {max_index}，词表大小: {tokenizer.vocab_size}")
            input_ids[input_ids >= tokenizer.vocab_size] = tokenizer.pad_token_id
            batch['input_ids'] = input_ids
            labels[labels >= tokenizer.vocab_size] = tokenizer.pad_token_id
            batch['labels'] = labels

        outputs = model(**batch)
        loss = outputs.loss
        print(f"批次 {i} 损失值: {loss.item()}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        progress_bar.update(1)

        del batch, outputs, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    progress_bar.close()

    # 模型评估
    print("切换到评估模式...")
    model.eval()
    inputs = tokenizer("你好啊", return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    print(f"测试输入的分词结果: {input_ids}")

    output = model.generate(
        input_ids,
        max_length=512,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
    )
    response = tokenizer.decode(
        output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    print(f"生成的回复: {response}")

main()
