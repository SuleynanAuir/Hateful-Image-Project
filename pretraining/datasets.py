import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPTokenizer, CLIPProcessor



class HatefulMemesDataset(Dataset):
    def __init__(self, root_folder, image_folder, split='train', image_pair='caption', image_size=224, use_top800=False):
        super(HatefulMemesDataset, self).__init__()
        self.root_folder = root_folder
        self.image_folder = image_folder
        self.split = split
        self.image_pair = image_pair
        self.image_size = image_size
        
        # 读取 CSV 并建立 ID 到数据的映射
        if use_top800:
            # 使用 top_800_info.csv (800张完整数据集)
            self.info_file = os.path.join(os.path.dirname(root_folder), 'sample_data', 'top_800_info.csv')
        else:
            # 使用 hateful_memes_expanded.csv (10000张完整数据集)
            self.info_file = os.path.join(root_folder, 'hateful_memes_expanded.csv')
        self.df = pd.read_csv(self.info_file)
        # 处理 split 映射: top_800 使用 'dev'，而 expanded 使用 'dev_seen'
        if use_top800 and self.split == 'dev_seen':
            self.split = 'dev'
        self.df = self.df[self.df['split']==self.split].reset_index(drop=True)
        float_cols = self.df.select_dtypes(float).columns
        self.df[float_cols] = self.df[float_cols].fillna(-1).astype('Int64')
        
        # 从图片文件夹获取所有图片，根据文件名反查 CSV
        self.image_files = []
        for fn in os.listdir(image_folder):
            if fn.endswith('.png'):
                # 从文件名提取 ID (去掉 _masked/_inpainted 等后缀)
                img_id = fn.replace('_masked.png', '.png').replace('_inpainted.png', '.png').replace('.png', '')
                # 在 CSV 中查找对应的行
                img_path = f"img/{img_id}.png"
                matching_rows = self.df[self.df['img'] == img_path]
                if not matching_rows.empty:
                    self.image_files.append((fn, matching_rows.iloc[0]))
        
        print(f"Found {len(self.image_files)} images with matching CSV entries for split={split}")
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        image_fn, row = self.image_files[idx]
        item = {}
        
        img_path = os.path.join(self.image_folder, image_fn)
        item['image'] = Image.open(img_path).convert('RGB').resize((self.image_size, self.image_size))
        item['text'] = row[self.image_pair]
        item['label'] = row['label']

        return item



class CustomCollator(object):

    def __init__(self, args):
        self.args = args
        self.image_processor = CLIPProcessor.from_pretrained(args.clip_pretrained_model)
        self.text_processor = CLIPTokenizer.from_pretrained(args.clip_pretrained_model)

    def __call__(self, batch):
        pixel_values = self.image_processor(images=[item['image'] for item in batch], return_tensors="pt")['pixel_values']
        text_output = self.text_processor([item['text'] for item in batch], padding=True, return_tensors="pt", truncation=True)
        labels = torch.LongTensor([item['label'] for item in batch])

        batch_new = {}
        batch_new['pixel_values'] = pixel_values,
        batch_new['input_ids'] = text_output['input_ids']
        batch_new['attention_mask'] = text_output['attention_mask']
        batch_new['labels'] = labels

        return batch_new



def load_dataset(args, split):

    # map to existing folders in this repo
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # 根据 dataset 参数决定是否使用 top_800
    use_top800 = args.dataset in ['masked', 'inpainted']
    
    if args.dataset == 'original':
        image_folder = os.path.join(project_root, 'data', 'img')
    elif args.dataset == 'masked':
        image_folder = os.path.join(project_root, 'data', 'sample_data', 'top_800_mask')
    elif args.dataset == 'inpainted':
        image_folder = os.path.join(project_root, 'data', 'sample_data', 'top_800_inpaint')

    dataset = HatefulMemesDataset(
        root_folder=os.path.join(project_root, 'data', 'hateful_memes'),
        image_folder=image_folder,
        split=split,
        image_pair=args.image_pair,
        image_size=args.image_size,
        use_top800=use_top800,
    )

    return dataset
