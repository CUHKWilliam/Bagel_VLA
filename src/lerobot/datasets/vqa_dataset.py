import torch
import os
import json
import imageio
import cv2
import pickle
from transformers import AutoProcessor



class VQADataset(torch.utils.data.Dataset):
    def __init__(self, repo_id, transform):
        self.root_path = repo_id
        data = []
        self.num_episodes = 0
        self.num_frames = 0
        for json_name in os.listdir(self.root_path):
            if json_name.endswith('json'):
                data += json.load(open(os.path.join(self.root_path, json_name), 'r'))
                meta_file_path = os.path.join(self.root_path, json_name.replace('.json', '_meta.pkl'))
                meta = pickle.load(open(meta_file_path, 'rb'))
                self.num_episodes += meta['num_episodes']
                self.num_frames += meta['num_frames']

        self.data = data
        self.transform = transform
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",use_fast=False)

    def __getitem__(self, idx):
        a_vqa_data = self.data[idx]
        item = {}
        video_path = os.path.join(self.root_path, a_vqa_data['metadata']['video_location'])
        convs = a_vqa_data['conversations']
        chat = self.processor.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
        vid = imageio.get_reader(video_path)
        image0 = vid.get_data(0)
        image1 = vid.get_data(vid.count_frames() - 1)
        image = cv2.hconcat([image0, image1])
        if self.transform is not None:
            image = self.transform(image)
        item['observation.images.image'] = image
        item['task'] = chat
        return item

    def __len__(self, ):
        return len(self.data)
    

class MultiVQADataset(torch.utils.data.Dataset):
    def __init__(self, repo_ids, transform):
        data = []
        root_paths = []
        self.num_episodes = 0
        self.num_frames = 0
        for repo_id in repo_ids:
            for json_name in os.listdir(repo_id):
                if json_name.endswith('json'):
                    a_data = json.load(open(os.path.join(repo_id, json_name), 'r'))
                    data += a_data
                    root_paths += [repo_id for _ in range(len(a_data))]
                    meta_file_name = os.path.join(os.path.join(repo_id, json_name.replace(".json", "_meta.pkl")))
                    meta = pickle.load(open(meta_file_name), 'rb')
                    self.num_episodes += meta['num_epidoes']
                    self.num_frames += meta['num_frames']
        self.data = data
        self.root_paths = root_paths
        self.transform = transform
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",use_fast=False)

    def __len__(self, ):
        return len(self.data)

    def __getitem__(self, idx):
        a_vqa_data = self.data[idx]
        root_path = self.root_paths[idx]
        item = {}
        video_path = os.path.join(root_path, a_vqa_data['metadata']['video_location'])
        convs = a_vqa_data['conversations']
        chat = self.processor.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
        image0 = vid.get_data(0)
        image1 = vid.get_data(vid.count_frames() - 1)
        image = cv2.hconcat([image0, image1])
        if self.transform is not None:
            image = self.transform(image)
        item['observation.images.image'] = image
        item['task'] = chat
        return item
    
       
