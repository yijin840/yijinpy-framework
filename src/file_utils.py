import pickle


def serilable_file(file_path):
    with open(file_path, "rb") as file:
        file_data = file.read()
        serialized_data = pickle.dumps(file_data)
    return serialized_data


def load_serilable_data(serialized_data):
    if serialized_data:
        return pickle.loads(serialized_data)  # 反序列化数据
    return None
