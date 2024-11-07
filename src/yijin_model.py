class YijinModel:

    def __init__(self, model_path):
        if model_path == None:
            self.model_path = self.createModel()
        # 模型路径
        self.model_path = model_path
        pass

    def loadDataStore(self):
        
        pass

    def train(self):
        pass

    def saveDataStore(self):
        pass

    def translate(self, text):
        print(f"text: {text}")
        pass

    def createDateStore(self):
        pass

    def createModel(self) -> str:
        return ""
    
    
