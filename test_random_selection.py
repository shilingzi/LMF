import os
import torch
from datasets import make
from torch.utils.data import DataLoader

def test_random_selection():
    # 使用当前目录下的图像进行测试
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建验证数据集
    val_dataset = make({
        'name': 'image-folder',
        'args': {
            'root_path': test_dir,
            'random_k': 3,  # 使用较小的数字进行测试
            'repeat': 1,
            'cache': 'in_memory'
        }
    })
    
    # 创建数据加载器
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # 打印选中的图像文件名
    print("Selected images in first run:")
    for i, data in enumerate(val_loader):
        print(f"Image {i+1}")
    
    # 创建另一个数据集实例，应该选择不同的图像
    val_dataset2 = make({
        'name': 'image-folder',
        'args': {
            'root_path': test_dir,
            'random_k': 3,
            'repeat': 1,
            'cache': 'in_memory'
        }
    })
    
    val_loader2 = DataLoader(val_dataset2, batch_size=1, shuffle=False)
    
    print("\nSelected images in second run:")
    for i, data in enumerate(val_loader2):
        print(f"Image {i+1}")

if __name__ == '__main__':
    test_random_selection() 