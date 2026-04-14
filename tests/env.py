import torch
print(torch.__version__)       # 看版本
print(torch.version.cuda)     # 看cuda → None就是CPU版
print(torch.cuda.is_available())