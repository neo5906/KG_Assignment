import torch
print("CUDA 是否可用:", torch.cuda.is_available())
print("CUDA 版本:", torch.version.cuda)
print("GPU 设备数量:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("当前 GPU 名称:", torch.cuda.get_device_name(0))