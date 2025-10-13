import torch
import os
import json
import imageio
import cv2
import pickle

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

    def __getitem__(self, idx):
        a_vqa_data = self.data[idx]
        item = {}
        if 'metadata' in a_vqa_data.keys() and 'video_location' in a_vqa_data['metadata'].keys():
            video_path = os.path.join(self.root_path, a_vqa_data['metadata']['video_location'])
            convs = a_vqa_data['conversations']
            vid = imageio.get_reader(video_path)
            image0 = vid.get_data(0)
            image1 = vid.get_data(vid.count_frames() - 1)
            image = cv2.hconcat([image0, image1])
        else:
            raise NotImplementedError
        if self.transform is not None:
            image = self.transform(image)
        image =  torch.from_numpy(image / 255.).permute((2, 0, 1))
        item['observation.images.image'] = image
        item['task'] = json.dumps(convs)
        return item

    def __len__(self, ):
        return len(self.data)
    

class MultiVQADataset(torch.utils.data.Dataset):
    def __init__(self, repo_ids, transform):
        data = []
        root_paths = []
        self.num_episodes = 0
        self.num_frames = 0
        self.datasets = []
        for repo_id in repo_ids:
            for json_name in os.listdir(repo_id):
                if json_name.endswith('json'):
                    root_paths += [repo_id for _ in range(len(a_data))]
                    meta_file_name = os.path.join(os.path.join(repo_id, json_name.replace(".json", "_meta.pkl")))
                    meta = pickle.load(open(meta_file_name), 'rb')
                    self.num_episodes += meta['num_epidoes']
                    self.num_frames += meta['num_frames']
                    self.datasets.append(
                        VQADataset(repo_id, transform)
                    )
        self.datasets = datasets
        self.root_paths = root_paths
        self.transform = transform
         

    def __len__(self, ):
        return len(self.data)

    def __getitem__(self, idx):
        dataset = np.random.choice(self.datasetes)
        item = dataset.__getiem__(idx)
        return item
    
       
