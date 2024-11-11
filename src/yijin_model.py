class YijinTranslateModel:

    def __init__(self, model_path):
        if model_path == None:
            self.model_path = self.create_model()
        # 模型路径
        self.model_path = model_path
        pass

    def load_data_store(self):

        pass

    def train(self):
        pass

    def save_data_store(self):
        pass

    def translate(self, text):
        print(f"text: {text}")
        pass

    def create_date_store(self):
        pass

    def create_model(self) -> str:
        return ""
