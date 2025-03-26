import torch

print('PyTorch版本:', torch.__version__)
print('CUDA是否可用:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('CUDA版本:', torch.version.cuda)
    print('GPU数量:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}:', torch.cuda.get_device_name(i))
else:
    print('CUDA不可用，使用CPU模式')

# 测试使用GPU进行简单运算
if torch.cuda.is_available():
    print("\n测试GPU计算...")
    a = torch.rand(10000, 10000).cuda()
    b = torch.rand(10000, 10000).cuda()
    
    # 记录开始时间
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    c = a @ b  # 矩阵乘法
    end.record()
    
    # 等待计算完成
    torch.cuda.synchronize()
    
    print(f"GPU矩阵乘法耗时: {start.elapsed_time(end):.2f} ms") 