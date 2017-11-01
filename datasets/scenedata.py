import os

import numpy as np 
from PIL import Image
from torch.utils import data
from torchvision import transforms
import json

train_img_path = '/home/wangshuo/experiment/compet/AIC/datasets/ai_challenger_scene_train_20170904/scene_train_images_20170904'
train_ann_path = '/home/wangshuo/experiment/compet/AIC/datasets/ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json'
val_img_path = '/home/wangshuo/experiment/compet/AIC/datasets/ai_challenger_scene_validation_20170908/scene_validation_images_20170908'
val_ann_path = '/home/wangshuo/experiment/compet/AIC/datasets/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json'
test_img_path = '/home/wangshuo/experiment/compet/AIC/datasets/ai_challenger_scene_test_a_20170922/scene_test_a_images_20170922'


def dataset_process(mode):
	assert mode in ['train', 'val', 'test']
	items = []

	if mode == 'train':
		with open(train_ann_path) as f:
			ann_dict = json.load(f)

		for it in ann_dict:
			img_path = os.path.join(train_img_path, it['image_id'])
			img_label = int(it['label_id'])
			item = (img_path, img_label)
			items.append(item)

	elif mode == 'val':
		with open(val_ann_path) as f:
			ann_dict = json.load(f)

		for it in ann_dict:
			img_path = os.path.join(val_img_path, it['image_id'])
			img_label = int(it['label_id'])
			item = (img_path, img_label)
			items.append(item)
	
	else:
		items = os.listdir(test_img_path)


	return items


class SceneDataset(data.Dataset):
	def __init__(self, mode, transform=None):
		self.items = dataset_process(mode)
		self.mode = mode
		self.transform = transform

	def __getitem__(self, index):
		if self.mode in ['train', 'val']:
			img_path, label_id = self.items[index]
			img = Image.open(img_path).convert('RGB')

			if self.transform is not None:
				img = self.transform(img)

			return img, label_id
		else:
                    img_id = self.items[index]
                    img_path = os.path.join(test_img_path, img_id)
                    img = Image.open(img_path).convert('RGB')
                    if self.transform is not None:
                        img = self.transform(img)
                        
                    return img,img_id

	def __len__(self):
		return len(self.items)



if __name__ == '__main__':
    pass
    #items = dataset_process('train')
    #print(items[0])


