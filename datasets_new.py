import os

import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from transformers import CLIPTokenizer, CLIPProcessor, AutoTokenizer

class HatefulMemesDataset(Dataset):
    def __init__(self, root_folder, image_folder, split='train', labels='original', image_size=224, use_top800=False):
        super(HatefulMemesDataset, self).__init__()
        self.root_folder = root_folder
        self.image_folder = image_folder
        self.split = split
        self.labels = labels
        self.image_size = image_size
        
        # 使用 top_800 或 expanded 文件
        if use_top800:
            self.info_file = os.path.join(os.path.dirname(root_folder), 'sample_data', 'top_800_new_info.csv')
            if self.split in ['dev_seen', 'eval']:
                self.split = 'dev'
        else:
            self.info_file = os.path.join(root_folder, 'hateful_memes_expanded.csv')
        
        # 加载 CSV
        self.df = pd.read_csv(self.info_file)
        
        # 处理 split 别名
        split_to_load = split
        if split in ['dev_seen', 'eval', 'test_seen']:
            split_to_load = 'dev'
        elif split == 'test_unseen':
            split_to_load = 'test'
        
        self.df = self.df[self.df['split'] == split_to_load].reset_index(drop=True)
        
        # 打印加载信息以便调试
        print(f"[{self.__class__.__name__}] Loaded split='{split}' (mapped to '{split_to_load}'): {len(self.df)} samples from {os.path.basename(self.info_file)}")

        # 修复 float 列
        float_cols = self.df.select_dtypes(float).columns
        self.df[float_cols] = self.df[float_cols].fillna(-1).astype('Int64')

        # fine-grained 标签处理
        if split in ['test_seen', 'test_unseen']:
            self.fine_grained_labels = []
        elif self.labels == 'fine_grained':
            self.pc_columns = [col for col in self.df.columns if col.endswith('_pc') and not col.endswith('_gold_pc')]
            self.pc_columns.remove('gold_pc')
            self.attack_columns = [col for col in self.df.columns if col.endswith('_attack') and not col.endswith('_gold_attack')]
            self.attack_columns.remove('gold_attack')
            self.fine_grained_labels = self.pc_columns + self.attack_columns
        elif self.labels == 'fine_grained_gold':
            self.pc_columns = [col for col in self.df.columns if col.endswith('_gold_pc')]
            self.attack_columns = [col for col in self.df.columns if col.endswith('_gold_attack')]
            self.fine_grained_labels = self.pc_columns + self.attack_columns
        else:
            self.fine_grained_labels = []
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = {}
        
        image_fn = row['img'].split('/')[1]
        
        # masked / inpaint 命名处理
        if 'top_800_mask' in self.image_folder:
            image_fn = image_fn.replace('.png', '_masked.png')
        elif 'top_800_inpaint' in self.image_folder:
            image_fn = image_fn.replace('.png', '_inpainted.png')
        
        item['image'] = Image.open(f"{self.image_folder}/{image_fn}").convert('RGB').resize(
            (self.image_size, self.image_size)
        )

        # 基本字段
        item['text'] = row['text']
        item['label'] = row['label']
        item['idx_meme'] = row['id']
        item['idx_image'] = row['pseudo_img_idx']
        item['idx_text'] = row['pseudo_text_idx']
        item['caption'] = row['caption'] if 'caption' in row else ''
        item['description'] = row['description'] if 'description' in row else ''
        item['keywords'] = row['keywords'] if 'keywords' in row else ''

        # fine-grained
        if self.labels.startswith('fine_grained'):
            for label in self.fine_grained_labels:
                item[label] = row[label]

        return item


class TamilMemesDataset(Dataset):
    def __init__(self, root_folder, split='train', image_size=224):
        """
        First, preprocess Tamil Troll Memes using `hateclipper/preprocessing/format_tamil_memes.ipynb`
        """
        super(TamilMemesDataset, self).__init__()
        self.root_folder = root_folder
        self.split = split
        self.image_size = image_size
        self.info_file = os.path.join(root_folder, 'labels.csv')
        self.df = pd.read_csv(self.info_file)
        self.df = self.df[self.df['split']==self.split].reset_index(drop=True)
        self.fine_grained_labels = []
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = {}
        item['image'] = Image.open(f"{self.root_folder}/{row['meme_path']}").convert('RGB').resize((self.image_size, self.image_size))
        item['text'] = row['text']
        item['caption'] = row['text_transliterated'] # named as caption just to match the format of HatefulMemesDataset
        item['label'] = row['is_troll']

        return item

class PropMemesDataset(Dataset):
    def __init__(self, root_folder, split='train', image_size=224):
        super(PropMemesDataset, self).__init__()
        self.root_folder = root_folder
        self.split = split
        self.image_size = image_size
        self.info_file = os.path.join(root_folder, f'annotations/{self.split}.jsonl')
        self.df = pd.read_json(self.info_file, lines=True)
        self.fine_grained_labels = ['Black-and-white Fallacy/Dictatorship', 'Name calling/Labeling', 'Smears', 'Reductio ad hitlerum', 'Transfer', 'Appeal to fear/prejudice', \
            'Loaded Language', 'Slogans', 'Causal Oversimplification', 'Glittering generalities (Virtue)', 'Flag-waving', "Misrepresentation of Someone's Position (Straw Man)", \
            'Exaggeration/Minimisation', 'Repetition', 'Appeal to (Strong) Emotions', 'Doubt', 'Obfuscation, Intentional vagueness, Confusion', 'Whataboutism', 'Thought-terminating cliché', \
            'Presenting Irrelevant Data (Red Herring)', 'Appeal to authority', 'Bandwagon']
        mlb = MultiLabelBinarizer().fit([self.fine_grained_labels])
        self.df = self.df.join(pd.DataFrame(mlb.transform(self.df['labels']),
                                            columns=mlb.classes_,
                                            index=self.df.index))
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = {}
        item['image'] = Image.open(f"{self.root_folder}/images/{row['image']}").convert('RGB').resize((self.image_size, self.image_size))
        item['text'] = " ".join(row['text'].replace("\n", " ").strip().lower().split())
        item['labels'] = row[self.fine_grained_labels].values.tolist()
        for label in self.fine_grained_labels:
            item[label] = row[label]

        return item

class CustomCollator(object):

    def __init__(self, args, fine_grained_labels, multilingual_tokenizer_path='none'):
        self.args = args
        self.fine_grained_labels = fine_grained_labels
        self.image_processor = CLIPProcessor.from_pretrained(args.clip_pretrained_model)
        self.text_processor = CLIPTokenizer.from_pretrained(args.clip_pretrained_model)
        if multilingual_tokenizer_path != 'none':
            self.text_processor = AutoTokenizer.from_pretrained(multilingual_tokenizer_path)

    def __call__(self, batch):

        pixel_values = self.image_processor(
            images=[item['image'] for item in batch],
            return_tensors="pt"
        )['pixel_values']

        enhanced_texts = []
        for item in batch:
            parts = []
            if item.get('text', '') != '':
                parts.append(item['text'])
            if item.get('caption', '') != '':
                parts.append(item['caption'])
            if item.get('description', '') != '':
                parts.append(item['description'])
            if item.get('keywords', '') != '':
                parts.append(item['keywords'])
            enhanced_texts.append(" [SEP] ".join(str(part) for part in parts))



        text_output = self.text_processor(
            enhanced_texts,
            padding=True,
            return_tensors="pt",
            truncation=True
        )

        if self.args.dataset in ['original', 'masked', 'inpainted', 'tamil']:
            caption_output = self.text_processor(
                [item['caption'] for item in batch],
                padding=True,
                return_tensors="pt",
                truncation=True
            )
            labels = torch.LongTensor([item['label'] for item in batch])

        if self.args.dataset in ['original', 'masked', 'inpainted']:
            idx_memes = torch.LongTensor([item['idx_meme'] for item in batch])
            idx_images = torch.LongTensor([item['idx_image'] for item in batch])
            idx_texts = torch.LongTensor([item['idx_text'] for item in batch])

        batch_new = {}
        batch_new['pixel_values'] = pixel_values
        batch_new['input_ids'] = text_output['input_ids']
        batch_new['attention_mask'] = text_output['attention_mask']

        if self.args.dataset in ['original', 'masked', 'inpainted', 'tamil']:
            batch_new['input_ids_caption'] = caption_output['input_ids']
            batch_new['attention_mask_caption'] = caption_output['attention_mask']
            batch_new['labels'] = labels

        if self.args.dataset in ['original', 'masked', 'inpainted']:
            batch_new['idx_memes'] = idx_memes
            batch_new['idx_images'] = idx_images
            batch_new['idx_texts'] = idx_texts

        if self.args.dataset in ['original', 'masked', 'inpainted', 'prop']:
            for label in self.fine_grained_labels:
                batch_new[label] = torch.LongTensor([item[label] for item in batch])

        if self.args.dataset == 'prop':
            batch_new['labels'] = torch.LongTensor([item['labels'] for item in batch])

        return batch_new



def load_dataset(args, split):

    # 根据数据集类型选择图片文件夹和CSV
    use_top800 = args.dataset in ['masked', 'inpainted']
    
    if args.dataset == 'original':
        image_folder = 'data/hateful_memes/img'
    elif args.dataset == 'masked':
        image_folder = 'data/sample_data/top_800_mask'
    elif args.dataset == 'inpainted':
        image_folder = 'data/sample_data/top_800_inpaint'
    
    if args.dataset == 'tamil':
        dataset = TamilMemesDataset(root_folder='data/Tamil_troll_memes', split=split, image_size=args.image_size)
    elif args.dataset == 'prop':
        dataset = PropMemesDataset(root_folder='data/propaganda-techniques-in-memes/data/datasets/propaganda/defaults', split=split, image_size=args.image_size)
    else:
        dataset = HatefulMemesDataset(root_folder='data/hateful_memes', image_folder=image_folder, split=split, 
            labels=args.labels, image_size=args.image_size, use_top800=use_top800)

    return dataset
