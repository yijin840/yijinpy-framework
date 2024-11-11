from torch import cuda, backends


def get_device():
    device = (
        "cuda"
        if cuda.is_available()
        else "mps" if backends.mps.is_available() else "cpu"
    )
    return device
