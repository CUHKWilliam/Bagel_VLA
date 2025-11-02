import torch
import os
import json
import imageio
import cv2
import pickle
import pandas
import numpy as np
from PIL import Image
import io

answer_templates = [
    "The correct choice is {}",
    "The right answer is {}",
    "The answer should be {}",
    "I believe the answer is {}",
    "Based on the options, the answer is {}",
    "After reviewing the choices, the answer is {}",
    "The best selection appears to be {}",
    "My selection is {}",
    "Option {} is correct",
    "Choice {} is the right answer",
    "The appropriate response is {}",
    "The accurate answer is {}",
    "The proper selection is {}",
    "Among the options, {} is correct",
    "The answer likely is {}",
    "The correct option is {}",
    "The valid choice is {}",
    "The answer corresponds to {}",
    "The right selection is {}",
    "After consideration, the answer is {}"
]
question_templates = [
    "What is depicted in this picture?",
    "Can you provide a description of this image?",
    "What do you see in this visual?",
    "How would you characterize this picture?",
    "What's shown in this photograph?",
    "Could you explain what this image contains?",
    "What appears in this picture?",
    "How would you describe what's in this image?",
    "What can you tell me about this visual content?",
    "What does this image show?",
    "Can you detail what's in this picture?",
    "What elements are present in this image?",
    "How would you summarize this photograph?",
    "What's the content of this image?",
    "Could you outline what this picture displays?",
    "What visual information does this contain?",
    "What's captured in this image?",
    "How would you portray this visual?",
    "What does this picture illustrate?",
    "Can you break down what's in this image?",
    "What's presented in this photograph?",
    "How would you characterize the contents of this picture?",
    "What details can you provide about this image?",
    "What's visible in this visual?",
    "Could you narrate what this image shows?",
    "What elements compose this picture?",
    "How would you depict what's in this image?",
    "What's the subject matter of this photograph?",
    "Can you elaborate on this image's content?",
    "What does this visual represent?",
    "What's featured in this picture?",
    "How would you explain this image?",
    "What components make up this visual?",
    "What's displayed in this photograph?",
    "Could you interpret this image for me?",
    "What's the scene in this picture?",
    "How would you summarize the visual elements?",
    "What content does this image contain?",
    "What can be observed in this picture?",
    "Can you give me an overview of this image?",
    "What's pictured here?",
    "How would you describe the visual composition?",
    "What does this image portray?",
    "What's the imagery in this picture?",
    "Could you walk me through what's in this image?",
    "What elements are captured in this visual?",
    "How would you account for what this picture shows?",
    "What's the visual content here?",
    "What does this photograph depict?",
    "Can you specify what's in this image?",
    "What's the view in this picture?",
    "How would you render a description of this visual?",
    "What's the subject of this image?",
    "What can you discern in this photograph?",
    "Could you give me a rundown of this picture?",
    "What's the pictorial content?",
    "How would you articulate what's in this image?",
    "What does this visual show?",
    "What's the imagery presented?",
    "Can you enumerate what's in this picture?",
    "What's the visual subject matter?",
    "How would you convey the contents of this image?",
    "What's the photographic content?",
    "What does this picture represent?",
    "Could you itemize what's visible in this visual?",
    "What's the scene depicted?",
    "How would you paraphrase what this image contains?",
    "What's the visual narrative?",
    "What elements are illustrated here?",
    "Can you catalog what's in this photograph?",
    "What's the picture showing?",
    "How would you report on this image's content?",
    "What's the visual story?",
    "What does this image exhibit?",
    "Could you list what's present in this picture?",
    "What's the imagery content?",
    "How would you state what's in this visual?",
    "What's the photographic subject?",
    "What can you identify in this image?",
    "Can you describe the visual elements?",
    "What's the composition of this picture?",
    "How would you express what this image shows?",
    "What's the depicted scene?",
    "What does this visual contain?",
    "Could you recount what's in this photograph?",
    "What's the picture about?",
    "How would you put into words what this image displays?",
    "What's the visual material?",
    "What elements are shown?",
    "Can you give a verbal account of this picture?",
    "What's the image presenting?",
    "How would you formulate a description of this visual?",
    "What's the content shown?",
    "What does this picture demonstrate?",
    "Could you provide a verbal depiction?",
    "What's the visual display?",
    "How would you word a description of this image?",
    "What's the imagery depicted?",
    "What can you see in this photograph?",
    "Can you articulate the contents of this picture?",
    "What's the visual presentation?",
    "How would you phrase what's in this image?",
    "What's the photographic representation?"
]
class VQADataset(torch.utils.data.Dataset):
    weight = 1.0
    ds_type = "vqa"
    def __init__(self, repo_id, transform):
        self.repo_id = repo_id
        self.root_path = repo_id
        data = None
        self.num_episodes = 0
        self.num_frames = 0
        for name in os.listdir(self.root_path):
            if name.endswith('json'):
                if data is None:
                    data = []
                data += json.load(open(os.path.join(self.root_path, name), 'r'))
                meta_file_path = os.path.join(sef.root_path, name.replace('.json', '_meta.pkl'))
            elif name.endswith('parquet'):
                a_data = pandas.read_parquet(os.path.join(self.root_path, name))
                meta_file_path = os.path.join(self.root_path, name.replace('.parquet', '_meta.pkl'))
                if data is None:
                    data = a_data
                else:
                    data = pandas.concat([data, a_data])
            elif name.endswith('jsonl'):
                data = open(os.path.join(self.root_path, name), 'r').readlines()
                meta_file_path = os.path.join(self.root_path, name.replace('.jsonl', '_meta.pkl'))
            else:
                continue
            meta = pickle.load(open(meta_file_path, 'rb'))
            self.num_episodes += meta['num_episodes']
            self.num_frames += meta['num_frames']
  
        self.data = data
        self.transform = transform
    
    def __getitem__(self, idx):
        if isinstance(self.data, pandas.DataFrame):
            a_vqa_data = dict(self.data.iloc[idx])
        elif isinstance(self.data, list):
            a_vqa_data = self.data[idx]
        else:
            raise NotImplementedError
        item = {}
        if isinstance(a_vqa_data, dict) and 'metadata' in a_vqa_data.keys() and 'video_location' in a_vqa_data['metadata'].keys():
            ## cosmos
            video_path = os.path.join(self.root_path, a_vqa_data['metadata']['video_location'])
            convs = a_vqa_data['conversations']
            vid = imageio.get_reader(video_path)
            image_indices = np.linspace(0, vid.count_frames() - 1, 5)
            images = []
            for image_index in image_indices:
                images.append(vid.get_data(image_index))
        elif isinstance(a_vqa_data, dict) and 'image_url' in a_vqa_data.keys():
            ## Capfusion
            image_dir = os.path.join(self.root_path, "images")
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, "{}.jpeg")
            # if os.path.exists(image_path):
            image = np.fromarray(Image.open(image_path).convert("RGB"))
            # else:
                ## TODO: for debug
                # image = np.zeros((100, 100, 3)).astype(np.uint8)
            images = [image]
            desc = a_vqa_data['capsfusion']
            convs = [
                        {"role": "user", 'content': [{'type': "text", 'text': np.random.choice(question_templates)}]},
                        {'role': "assistant", "content": [{'type': "text", "text": desc}]}
                    ]
        elif isinstance(a_vqa_data, dict) and 'image' in a_vqa_data.keys() and 'bytes' in a_vqa_data['image']:
            ## robo2vlm
            image = np.asarray(Image.open(io.BytesIO(a_vqa_data['image']['bytes'])).convert("RGB"))
            images = [image]
            choices_text = a_vqa_data['choices']
            ans = eval(choices_text)[a_vqa_data['correct_answer']]
            convs = [
                    {'role': "user", "content": [{'type': "text", "text": "{}. Choices: {}".format(a_vqa_data['question'], choices_text)}]},
                    {'role': "assistant", "content": [{"type": "text", "text": np.random.choice(answer_templates).format(ans)}]}
            ]
        elif isinstance(a_vqa_data, str):
            ## cambrian
            a_vqa_data = json.loads(a_vqa_data)
            if 'image' in a_vqa_data.keys() and a_vqa_data['image'] != "" and a_vqa_data['image'] is not None:
                image_path = os.path.join(self.root_path, a_vqa_data['image'])
                image = np.asarray(Image.open(image_path).convert("RGB"))
                images = [image]
                # images = []
            else:
                images = []
            orig_conv = a_vqa_data['conversations']
            convs = []
            for a_orig_conv in orig_conv:
                if a_orig_conv['from'] == "human":
                    convs.append({'role': "user", "content": [{'type': 'text', "text": a_orig_conv['value']}]})
                elif a_orig_conv['from'] == 'gpt':
                    convs.append({'role': 'assistant', 'content': [{'type': 'text', 'text': a_orig_conv['value']}]})
        else:
            raise NotImplementedError
        if self.transform is not None:
            for idx in range(len(images)):
                images[idx] = self.transform(images[idx])
        for idx in range(len(images)):
            image = images[idx]
            image =  torch.from_numpy(image / 255.).permute((2, 0, 1))
            item['vqa.images.image.{}'.format(idx)] = image
        item['task'] = json.dumps(convs)
        return item

    def __len__(self, ):
        return len(self.data)
    

class MultiVQADataset(torch.utils.data.Dataset):
    weight = 1.0
    def __init__(self, repo_ids, transform):
        data = []
        root_paths = []
        self.num_episodes = 0
        self.num_frames = 0
        self.datasets = []
        for repo_id in repo_ids:
            for name in os.listdir(repo_id):
                if name.endswith('json'):
                    root_paths += [repo_id for _ in range(len(a_data))]
                    meta_file_name = os.path.join(os.path.join(repo_id, name.replace(".json", "_meta.pkl")))
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
    
       
