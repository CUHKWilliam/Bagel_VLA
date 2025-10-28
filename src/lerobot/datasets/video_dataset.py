import torch
import os
import json
import imageio
import cv2
import pickle
import imageio.v3 as iio
import numpy as np

question_templates = [
    "The task is {}. Please predict the image for the next moment",
    "Given the task {}, forecast the subsequent image",
    "For the task {}, predict what the image will look like next",
    "Based on the task {}, anticipate the following image",
    "The objective is {}. Predict the next image frame",
    "With the task being {}, project the upcoming image",
    "Task: {}. Generate a prediction for the next visual frame",
    "Considering the task {}, foresee the next image",
    "Your task is {}. Predict the image at the next timestep",
    "For task {}, estimate what comes next visually",
    "The assignment is {}. Forecast the subsequent visual",
    "Given assignment {}, predict the following image",
    "Mission: {}. Anticipate the next image frame",
    "The challenge is {}. Project the image for the next moment",
    "Task description: {}. Predict the upcoming image",
    "With the objective {}, envision the next visual frame",
    "The goal is {}. Generate the next image prediction",
    "For the objective {}, predict the subsequent frame",
    "Your mission is {}. Forecast what the image will be next",
    "Assignment: {}. Anticipate the visual at the next moment",
    "The problem is {}. Predict the following image",
    "Given problem {}, project the next image frame",
    "Task defined as {}. Foresee the upcoming visual",
    "The target is {}. Estimate the next image",
    "For target {}, predict the image that comes next",
    "Objective: {}. Generate prediction for next visual moment",
    "With task {}, anticipate the subsequent image",
    "The job is {}. Forecast the next frame",
    "Job: {}. Predict what the image shows next",
    "Your job is {}. Project the following image",
    "The work is {}. Envision the next visual frame",
    "Work task: {}. Predict the image at next time step",
    "Given work {}, forecast the upcoming image",
    "The duty is {}. Anticipate what comes next visually",
    "Duty: {}. Predict the subsequent visual frame",
    "Your duty is {}. Generate next image prediction",
    "The assignment task is {}. Foresee the next image",
    "Task specification: {}. Estimate the following visual",
    "With specification {}, predict the next moment's image",
    "The defined task is {}. Project the upcoming frame",
    "Defined as {}, anticipate the next visual",
    "Your objective is {}. Forecast the image that follows",
    "Objective stated: {}. Predict the subsequent moment's image",
    "The mission objective is {}. Generate next visual prediction",
    "Mission brief: {}. Anticipate the following frame",
    "Challenge: {}. Predict what the image will be next",
    "The challenge task is {}. Project the next visual",
    "Problem statement: {}. Envision the upcoming image",
    "Given the problem {}, forecast the next image frame",
    "Target objective: {}. Predict the visual at next moment",
    "The goal task is {}. Estimate the subsequent image",
    "Goal: {}. Generate prediction for following visual",
    "Your target is {}. Foresee the next image frame",
    "Assignment objective: {}. Anticipate the upcoming visual",
    "Work objective: {}. Predict the next time step image",
    "The job task is {}. Project what comes next visually",
    "Job description: {}. Forecast the subsequent visual",
    "Duty assignment: {}. Predict the following frame",
    "The work assignment is {}. Envision the next image",
    "Task overview: {}. Generate next moment image prediction",
    "With the mission {}, anticipate the subsequent visual",
    "Mission parameters: {}. Predict the upcoming frame",
    "Challenge details: {}. Estimate the next visual moment",
    "Problem parameters: {}. Project the following image",
    "Target parameters: {}. Foresee the image that comes next",
    "Goal parameters: {}. Predict the visual for next timestep",
    "Objective parameters: {}. Anticipate the next image frame",
    "Assignment parameters: {}. Generate subsequent visual prediction",
    "Job parameters: {}. Forecast the image at next moment",
    "Work parameters: {}. Predict what follows visually",
    "Duty parameters: {}. Project the upcoming visual frame",
    "The specified task is {}. Envision the next image",
    "Specified as {}, predict the following visual",
    "Your specified task is {}. Anticipate the next frame",
    "Task instruction: {}. Generate prediction for next image",
    "Instruction: {}. Forecast the subsequent visual moment",
    "Following task {}, predict the upcoming image",
    "Based on assignment {}, project the next visual",
    "According to task {}, foresee the following frame",
    "Per the task {}, estimate the next image",
    "As per task {}, predict what the visual shows next",
    "In accordance with task {}, anticipate the subsequent image",
    "Following the objective {}, generate next visual prediction",
    "Based on objective {}, forecast the next frame",
    "According to objective {}, predict the upcoming visual",
    "Per the objective {}, project the following image",
    "As per objective {}, envision the next visual moment",
    "In line with task {}, anticipate what comes next",
    "Consistent with task {}, predict the next image",
    "Aligned with task {}, forecast the subsequent visual",
    "Corresponding to task {}, generate next frame prediction",
    "Relative to task {}, predict the following visual",
    "Pertaining to task {}, anticipate the upcoming image",
    "Regarding task {}, project the next visual frame",
    "With respect to task {}, foresee what the image will be next",
    "In reference to task {}, estimate the subsequent image",
    "Concerning task {}, predict the visual at next moment",
    "On the subject of task {}, generate next image forecast",
    "About the task {}, anticipate the following visual",
    "Task context: {}. Predict the upcoming frame",
    "Within context {}, project the next image",
    "Context: {}. Envision the subsequent visual"
]
class VideoDataset(torch.utils.data.Dataset):
    weight = 1.0
    ds_type = "video"
    def __init__(self, repo_id, transform):
        self.repo_id = repo_id
        self.root_path = repo_id
        data = []
        self.num_episodes = 0
        self.num_frames = 0
        for file_name in os.listdir(self.root_path):
            if file_name.endswith('json'):
                ## for something-something-v2 and ego4d
                data += json.load(open(os.path.join(self.root_path, file_name), 'r'))
                meta_file_path = os.path.join(self.root_path, file_name.replace('.json', '_meta.pkl'))
                meta = pickle.load(open(meta_file_path, 'rb'))
                self.num_episodes += meta['num_episodes']
                self.num_frames += meta['num_frames']
        self.data = data
        self.transform = transform
        
    def __getitem__(self, idx):
        a_video_data = self.data[idx]
        item = {}
        if 'id' in a_video_data:
            ## for somethiing-something-v2
            video_path = os.path.join(self.root_path, "videos", '{}.webm'.format(a_video_data['id']))
            frames_iter = iio.imiter(video_path)
            frames = []
            for frame in frames_iter:
                frames.append(frame)
            image = frames[0]
            num_frames = len(frames)
            next_image = frames[np.random.randint(low=int(0.5*num_frames), high=int(num_frames))]
            convs = [
                {
                    "role": "user",
                    "content": [{'type': 'text', 'text': np.random.choice(question_templates).format("\""+a_video_data['label']+"\"")}]
                }
            ]
        elif 'image' in a_video_data:
            ## for ego4d-video
            frames = np.load(os.path.join(self.root_path, a_video_data['image']))
            image = frames[0].transpose((1, 2, 0)) * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406] )
            image = np.clip(image, a_min=0, a_max=1) * 255
            num_frames = len(frames)
            next_image = frames[np.random.randint(low=int(0.5*num_frames), high=int(num_frames))]
            next_image = next_image.transpose((1, 2, 0)) * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406] )
            next_image = np.clip(next_image,  a_min=0, a_max=1) * 255
            convs = [
                {
                    "role": "user",
                    "content": [{'type': 'text', 'text': np.random.choice(question_templates).format("\""+a_video_data['caption']+"\"")}]
                }
            ]
        else:
            raise NotImplementedError
        if self.transform is not None:
            image = self.transform(image)
            next_image = self.transform(next_image)
        image =  torch.from_numpy(image / 255.).permute((2, 0, 1))
        next_image = torch.from_numpy(next_image / 255.).permute((2, 0, 1))
        item['actionless_video.images.image'] = image
        item['next.images.image'] = next_image
        item['task'] = json.dumps(convs)
        return item

    def __len__(self, ):
        return len(self.data)
    

class MultiVideoDataset(torch.utils.data.Dataset):
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
    
       
