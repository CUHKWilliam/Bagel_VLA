import torch
import os
import json
import imageio
import cv2


class VQADataset(torch.utils.data.Dataset):
    def __init__(self, repo_id, transform):
        self.root_path = repo_id
        data = []
        for json_name in os.walk(self.root_path):
            if json_name.endswith('json'):
                data += json.load(open(os.path.join(self.root_path, json_name), 'r'))
        self.data = data
        self.transform = transform
    def __getitem__(self, idx):
        a_vqa_data = self.data[idx]
        item = {}
        video_path = os.path.join(self.root_path, a_vqa_data['metadata']['video_location'])
        convs = a_vqa_data['conversations']
        chat = processor.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
        image0 = vid.get_data(0)
        image1 = vid.get_data(vid.count_frames() - 1)
        image = cv2.hconcat([image0, image1])
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
        for repo_id in repo_ids:
            for json_name in os.walk(repo_id):
                if json_name.endswith('json'):
                    a_data = json.load(open(os.path.join(repo_id, json_name), 'r'))
                    data += a_data
                    root_paths += [repo_id for _ in range(len(a_data))]
        self.data = data
        self.root_paths = root_paths
        self.transform = transform

    def __len__(self, ):
        return len(self.data)

    def __getitem__(self, idx):
        a_vqa_data = self.data[idx]
        root_path = self.root_paths[idx]
        item = {}
        video_path = os.path.join(root_path, a_vqa_data['metadata']['video_location'])
        convs = a_vqa_data['conversations']
        chat = processor.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
        image0 = vid.get_data(0)
        image1 = vid.get_data(vid.count_frames() - 1)
        image = cv2.hconcat([image0, image1])
        image = self.transform(image)
        item['observation.images.image'] = image
        item['task'] = chat
        return item
