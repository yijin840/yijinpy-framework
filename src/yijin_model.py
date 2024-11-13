import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, AdamW
from datasets import load_dataset
import torch.optim as optim


class YijinTranslateModel:

    def __init__(self):
        print("yijin translate model init.")
        pass

    ## 数据预处理
    def pre_process_data(self):
        pass

    ## 构建词汇表
    def build_vocabulary(self):
        pass

    ## 数据加载
    def load_data_store(self):
        pass

    ## 训练
    def train(self):
        pass

    ##保存数据
    def save_data_store(self):
        pass

    ## 创建数据
    def create_data_store(self):
        pass

    ## 创建模型
    def create_model(self):
        pass

    ## 加载模型
    def load_model(self):
        pass

    ## 导入模型
    def save_model(self):
        pass


## 对话模型 基于GPT-2 训练
class YijinGptModel:
    """
    1. 初始化参数
    2. 加载模型，如果没有就创建
    3. 加载数据集，没有就下载
    4. 训练
    5. 保存模型 or 训练数据
    6. 交互
    """

    def __init__(self, data_store=None):
        print("init yijin gpt model.")
        self.model = self.create_default_model()
        self.data_store = (
            self.load_data_store(data_store) or self.load_default_dataset()
        )
        self.tokenizer = self.get_default_tokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def create_default_model(self):
        return GPT2LMHeadModel.from_pretrained("gpt2")

    def tokenize_function(self, ds):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        return tokenizer(
            ds["text"], padding="max_length", truncation=True, max_length=512
        )

    def load_default_dataset(self):
        # dataset = load_dataset("roycehe/tieba")
        # self.train_dataset = dataset["train"].map(self.tokenize_function, batched=True)
        # self.validation_dataset = dataset["validation"].map(
        #     self.tokenize_function, batched=True
        # )
        # self.test_dataset = dataset["test"].map(self.tokenize_function, batched=True)
        # print(dataset)
        # return dataset
        # # return load_dataset(
        # # "text",
        # # data_files="https://huggingface.co/datasets/lhoestq/test/resolve/main/some_text.txt",
        # # )
        return load_dataset("daily_dialog", trust_remote_code=True)

    def get_default_tokenizer(self):
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        return tokenizer

    def train(self, epochs=3, batch_size=4, learning_rate=5e-5, max_length=512):
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")  # 打印当前的 epoch
            epoch_loss = 0  # 用于记录当前 epoch 的总损失

            # 加载训练数据
            inputs_data = self.data_store["train"]["dialog"]  # 保留原始数据

            for i in range(0, len(inputs_data), batch_size):
                batch = [" ".join(dialog) for dialog in inputs_data[i : i + batch_size]]
                # print(f"Batch: {batch}")  # 打印 batch 确认格式

                # 将输入转换为模型可处理的格式
                tokenized_inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,  # 设置最大长度
                    is_split_into_words=False,
                )
                # print(
                # f"Tokenized Inputs: {tokenized_inputs}"
                # )  # 打印 tokenized_inputs 查看返回值的结构
                tokenized_inputs = {
                    key: value.to(self.device)
                    for key, value in tokenized_inputs.items()
                }

                # 检查是否有超出词汇表范围的 token
                input_ids = tokenized_inputs["input_ids"]
                if input_ids.max().item() >= self.model.config.vocab_size:
                    print("Warning: Detected out-of-vocabulary token in input_ids")
                    continue  # 跳过这个 batch

                optimizer.zero_grad()
                # 直接传递字典，不需要解包
                outputs = self.model(
                    **tokenized_inputs, labels=tokenized_inputs["input_ids"]
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()  # 累加当前 batch 的损失

            # 打印每个 epoch 的平均损失
            print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(inputs_data)}")
            self.model.train()

    # 交互式对话生成
    def generate_response(self, prompt, max_length=50):
        # 切换到评估模式
        self.model.eval()
        # 使用 GPT-2 模型生成对话响应
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs["input_ids"].to(self.device)
        # 生成回复
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
        )

        # 解码并返回生成的响应
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    def save_data_store(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def load_data_store(self, model_path):
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
