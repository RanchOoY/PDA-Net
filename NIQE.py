import os
import cv2
import torch
import numpy as np

from tqdm import tqdm
from torchvision import transforms
from PIL import Image

import pyiqa

# ==========================================
# 导入你的模型
# ==========================================
from model_best import LYT   # 改成你的文件名


# ==========================================
# 配置
# ==========================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_PATH = 'PNA_Net_best_LOLv1.pth'

DATASETS = {
    'DICM': 'E:/data_1/DICM',
    'LIME': 'E:/data_1/LIME',
    'MEF': 'E:/data_1/MEF',
    'NPE': 'E:/data_1/NPE',
    'VV': 'E:/data_1/VV'
}

SAVE_ROOT = './results/NIQE'


# ==========================================
# 加载模型
# ==========================================
print('Loading model...')

model = LYT(filters=64).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# 兼容多种保存方式
if 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

print('Model loaded.')


# ==========================================
# NIQE指标
# ==========================================
niqe_metric = pyiqa.create_metric('niqe').to(DEVICE)


# ==========================================
# 图像预处理
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor()
])


# ==========================================
# 推理函数
# ==========================================
@torch.no_grad()
def enhance_image(img_path, max_size=768):

    img = Image.open(img_path).convert('RGB')

    w, h = img.size

    # ==========================================
    # Resize 防止OOM
    # ==========================================
    if max(h, w) > max_size:

        if h > w:
            new_h = max_size
            new_w = int(w * max_size / h)
        else:
            new_w = max_size
            new_h = int(h * max_size / w)

        # 保证8倍数（Attention更稳定）
        new_h = (new_h // 8) * 8
        new_w = (new_w // 8) * 8

        img = img.resize((new_w, new_h), Image.BICUBIC)

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    output = model(img_tensor)

    output = torch.clamp(output, 0, 1)

    return output


# ==========================================
# 保存图像
# ==========================================
def save_image(tensor, save_path):

    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    img = (img * 255.0).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path, img)


# ==========================================
# 测试单个数据集
# ==========================================
def test_dataset(dataset_name, dataset_path):

    print(f'\nTesting {dataset_name}...')

    save_dir = os.path.join(SAVE_ROOT, dataset_name)

    os.makedirs(save_dir, exist_ok=True)

    image_list = sorted(os.listdir(dataset_path))

    niqe_scores = []

    for img_name in tqdm(image_list):

        img_path = os.path.join(dataset_path, img_name)

        try:

            # 推理
            enhanced = enhance_image(img_path)

            # 保存结果
            save_path = os.path.join(save_dir, img_name)

            save_image(enhanced, save_path)

            # 计算 NIQE
            score = niqe_metric(enhanced).item()

            niqe_scores.append(score)


        except Exception as e:

            print(f'Error processing {img_name}: {e}')

        torch.cuda.empty_cache()

    avg_niqe = np.mean(niqe_scores)

    print(f'{dataset_name} Average NIQE: {avg_niqe:.4f}')

    return avg_niqe


# ==========================================
# 主函数
# ==========================================
def main():

    final_results = {}

    for dataset_name, dataset_path in DATASETS.items():

        score = test_dataset(dataset_name, dataset_path)

        final_results[dataset_name] = score

    print('\n==============================')
    print('Final NIQE Results')
    print('==============================')

    for k, v in final_results.items():

        print(f'{k}: {v:.4f}')


if __name__ == '__main__':
    main()