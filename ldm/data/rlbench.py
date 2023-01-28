import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


import glob
import numpy as np
import torch
import torchvision
import pickle
from panopticapi.utils import rgb2id
import ipdb
st = ipdb.set_trace
from PIL import Image
from torch.utils.data import Dataset
import random

class RlBenchBase(Dataset):
    def __init__(
        self,
        dataset_type,
        dataset,
        data_dir,
        config,
        # transform,
    ):
        super(RlBenchBase, self).__init__()
        fix_suport_cam = False
        overfit=False        
        # st()
        # all_tasks = glob.glob(data_dir +"/*" )
        self.taskset = pickle.load(open(f'{data_dir}/{dataset}',"rb"))
        self.dataset_type = dataset_type
        self.all_tasks = list(self.taskset.keys())

        # st()

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((config['size'],config['size'])),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    
        self.fix_suport_cam = fix_suport_cam
        self.data_dir = data_dir

        self.camera_views = ["front_rgb","left_shoulder_rgb","right_shoulder_rgb","overhead_rgb"]

        self.overfit = overfit
        # self.transform = transform

    def __len__(self):
        return 200000

    def __getitem__(self, idx):
        idx %= len(self.all_tasks)
        if  self.overfit:
            idx = 0
        task_valk = self.all_tasks[idx]
        # st()
        example = {}

        if self.dataset_type == "intra_variations":
            all_episodes = list(self.taskset[task_valk].keys())
            if len(all_episodes) == 1:
                episodes = [all_episodes[0],all_episodes[0]]
            else:
                episodes = random.sample(all_episodes,2)
            random.shuffle(episodes)
            support_episode, target_episode = episodes
            support_episode_dir, support_episode_name = support_episode.split("/")
            target_episode_dir, target_episode_name = target_episode.split("/")

            task_folder, variation_folder = task_valk.split("/")

            support_folder_name = f"{self.data_dir}/{support_episode_dir}/{task_folder}/{variation_folder}/episodes/{support_episode_name}/"
            target_folder_name = f"{self.data_dir}/{target_episode_dir}/{task_folder}/{variation_folder}/episodes/{target_episode_name}/"
            support_len = self.taskset[task_valk][support_episode]
            target_len = self.taskset[task_valk][target_episode]
            support_gap  = support_len//5
            target_gap  = target_len//5
            support_frames_idx = [random.randint(support_gap*i,support_gap*(i+1)-1) for i in range(5)]
            target_frames_idx = [random.randint(target_gap*i,target_gap*(i+1)-1) for i in range(5)]
            target_frames_idx  = random.sample(target_frames_idx,2)
            target_frames_idx.sort()
            # st()
            camera_view_support, camera_view_target  = random.sample(self.camera_views,2)

            if self.fix_suport_cam:
                camera_view_support = "front_rgb"
            # st()
            support_rgb = [self.transform(Image.open(f"{support_folder_name}/{camera_view_support}/{idx}.png").convert('RGB')) for idx in support_frames_idx]
            target_rgb = [self.transform(Image.open(f"{target_folder_name}/{camera_view_target}/{idx}.png").convert('RGB')) for idx in target_frames_idx]
            support_rgb_stack = torch.stack(support_rgb)
            target_rgb_stack = torch.stack(target_rgb)
            support_idxs = torch.tensor(support_frames_idx).float()/support_len
            target_idxs = torch.tensor(target_frames_idx).float()/target_len
            # st()

            # return support_rgb_stack, target_rgb_stack, support_idxs, target_idxs, target_len
        elif self.dataset_type == "intra_views":
            # st()
            episode_info = random.choice(list(self.taskset[task_valk].keys()))
            task_folder, variation_folder = task_valk.split("/")

            episode_dir,episode_name = episode_info.split("/")
            # task_desc = "_".join(task_valk.split("_")[:-1])
            variation_pickled_file = f"{self.data_dir}/{episode_dir}/{task_folder}/{variation_folder}/variation_descriptions.pkl"
            all_task_desc = pickle.load(open(variation_pickled_file,"rb"))
            task_desc = random.sample(all_task_desc,1)


            folder_name =  f"{self.data_dir}/{episode_dir}/{task_folder}/{variation_folder}/episodes/{episode_name}/"
            num_ex = self.taskset[task_valk][episode_info]

            if  self.overfit:
                later_idx = 111
                camera_view_support, camera_view_target = ("front_rgb","left_shoulder_rgb")
            else:
                start_iter = 0
                later_idx = random.randint(start_iter,num_ex-1)
                camera_view_support, camera_view_target  = random.sample(self.camera_views,2)
            init_idx = 0

            # support_init_rgb = Image.open(f"{folder_name}/{camera_view_support}/{init_idx}.png").convert('RGB')
            support_later_rgb = Image.open(f"{folder_name}/{camera_view_support}/{later_idx}.png").convert('RGB')

            target_init_rgb = Image.open(f"{folder_name}/{camera_view_target}/{init_idx}.png").convert('RGB')        
            target_later_rgb = Image.open(f"{folder_name}/{camera_view_target}/{later_idx}.png").convert('RGB')
            # st()
            max_frames = 2.0

            timestep =  torch.arange(1,2)/max_frames

            # source_frame 
            # st()
            support_folder_name_new = f"{folder_name}/{camera_view_support}"
            target_folder_name_new = f"{folder_name}/{camera_view_target}"
            # st()
            support_later_rgb, target_init_rgb, target_later_rgb, timestep,support_folder_name_new, target_folder_name_new, num_ex = self.normalize(support_later_rgb), self.normalize(target_init_rgb), self.normalize(target_later_rgb), timestep,support_folder_name_new, target_folder_name_new, num_ex
            # st()

            example["support_images"] = (torch.cat([support_later_rgb,target_init_rgb],axis=0).permute(1,2,0)*2.0) -1.0
            example["caption"] = task_desc[0]
            example["image"] = (target_later_rgb.permute(1,2,0)*2.0) -1.0
            # example["caption"] = "hello world"        
            # return (support_init_rgb,support_later_rgb, target_init_rgb, target_later_rgb, timestep,support_folder_name_new, target_folder_name_new, num_ex)

        elif self.dataset_type == "intra_v_episodes":
            all_episodes = list(self.taskset[task_valk].keys())
            if len(all_episodes) == 1:
                episodes = [all_episodes[0],all_episodes[0]]
            else:
                episodes = random.sample(all_episodes,2)
            random.shuffle(episodes)
            support_episode, target_episode = episodes
            support_episode_dir, support_episode_name = support_episode.split("/")
            target_episode_dir, target_episode_name = target_episode.split("/")

            task_folder, variation_folder = task_valk.split("/")

            support_folder_name = f"{self.data_dir}/{support_episode_dir}/{task_folder}/{variation_folder}/episodes/{support_episode_name}/"
            target_folder_name = f"{self.data_dir}/{target_episode_dir}/{task_folder}/{variation_folder}/episodes/{target_episode_name}/"
            support_len = self.taskset[task_valk][support_episode]
            target_len = self.taskset[task_valk][target_episode]


            camera_view_support, camera_view_target  = random.sample(self.camera_views,2)            
            support_folder_name = f"{support_folder_name}/{camera_view_support}"
            target_folder_name = f"{target_folder_name}/{camera_view_target}"
            # return support_folder_name, target_folder_name, support_len, target_len
        # st()
        return example





class RLBenchTrain(RlBenchBase):
    def __init__(self, dataset_type='intra_views', dataset='train_val_dict.pkl', data_dir='/projects/katefgroup/rlbench_custom/', **kwargs):
        # st()
        super().__init__(dataset_type=dataset_type, dataset=dataset, data_dir=data_dir,**kwargs)


class RLBenchValidation(RlBenchBase):
    def __init__(self, dataset_type='intra_views', dataset='test_val_dict.pkl', data_dir='/projects/katefgroup/rlbench_custom/', **kwargs):
        # st()
        super().__init__(dataset_type=dataset_type, dataset=dataset, data_dir=data_dir,**kwargs)
