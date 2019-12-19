# -*- coding:UTF-8 -*-
import os
import sys
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np


sys.path.append("..")

from utils.augmentation import SSDAugmentation

##https://blog.csdn.net/gzj2013/article/details/82421166
##https://blog.csdn.net/u013832707/article/details/94445495

#HOME = os.path.expanduser("~")
HOME = '/mnt/sdb/elliot/yolo-v3'

COCO_ROOT = os.path.join(HOME,'data/coco')
IMAGES = 'images'
ANNOTATIONS = 'annotations'
INSTANCES = 'instances_{}.json'

COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush')



def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0),targets



def get_label_map(label_file):
    label_map = {}
    labels = open(label_file,'r')
    for line in labels:
        ids = line.split(',')
        label_map[int(ids[0])] = int(ids[1])

    return label_map



class COCOAnnotationTransform(object):

    def __init__(self):
        self.label_map = get_label_map(os.path.join(COCO_ROOT,'coco_labels.txt'))

    def __call__(self, target,width,height):

        scale = np.array([width,height,width,height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                label_idx = self.label_map[obj['category_id']] - 1
                final_box = list(np.array(bbox)/scale)
                final_box.append(label_idx)
                res += [final_box] # [xmin, ymin, xmax, ymax, label_idx]


        return res # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class COCODetection(data.Dataset):

    def __init__(self,root,image_set='train2014',transform=None,target_transform=COCOAnnotationTransform(),dataset_name='MS COCO'):
        # sys.path.append(os.path.join(root,COCO_API))
        from pycocotools.coco import COCO

        self.root = os.path.join(root,IMAGES,image_set)
        self.coco = COCO(os.path.join(root,ANNOTATIONS,INSTANCES.format(image_set)))

        self.ids = list(self.coco.imgToAnns.keys()) ##一个dict  图片的id：[annotation1,annotation2......]
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name

    def __getitem__(self, index):

        im,gt,h,w = self.pull_item(index)
        print(im.shape,"..",gt.shape)
        return im,gt

    def __len__(self):
        return len(self.ids)


    def pull_item(self,index):
        img_id = self.ids[index]
        target = self.coco.imgToAnns[img_id]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)

        target = self.coco.loadAnns(ann_ids)
        path = os.path.join(self.root,self.coco.loadImgs(img_id)[0]['file_name'])
        assert os.path.exists(path)
        img = cv2.imread(os.path.join(self.root,path))

        height,width,_ = img.shape
        if self.target_transform is not None:
            target = self.target_transform(target,width,height)
        if self.transform is not None:
            target = np.array(target)
            img,boxes,labels = self.transform(img,target[:,:4],target[:,4])

            img = img[:,:,(2,1,0)]
            target = np.hstack((boxes,np.expand_dims(labels,axis=1)))

        return torch.from_numpy(img).permute(2,0,1),target,height,width


    def pull_image(self,index):

        img_id = self.ids[index]

        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(os.path.join(self.root,path),cv2.IMREAD_COLOR)

    def pull_anno(self,index):

        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)

        return self.coco.loadAnns(ann_ids)

    # def __repr__(self):
    #     fmt_str = 'Dataset' + self.__class__.__name__ + '\n'
    #     fmt_str += ' Number of datapoint: {}\n'.format(self.__len__())
    #     fmt_str += ' Root Location:{}\n'.format(self.root)
    #     tmp = '  Transforms (if any): '
    #     fmt_str += '{0}{1}\n'.format(tmp,self.transform.__repr__().replace('\n','\n'+' '*len(tmp)))
    #     tmp = ' Target Transforms (if any)'
    #     fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n','\n'+' '*len(tmp)))
    #     return fmt_str


def get_COCODataLoader(batch_size,num_workers):
    MEANS = (104, 117, 123)
    dataset = COCODetection(root=COCO_ROOT, transform=SSDAugmentation(300, MEANS))
    data_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True,
                                  collate_fn=detection_collate,
                                  # pin_memory=True
                                  )
    return data_loader



if __name__ == '__main__':
    MEANS = (104, 117, 123)
    dataset = COCODetection(root=COCO_ROOT,transform=SSDAugmentation(300,MEANS))
    data_loader = data.DataLoader(dataset,
                                  batch_size=32,
                                  #num_workers=args.num_workers,
                                  shuffle=True,
                                  collate_fn=detection_collate,
                                  #pin_memory=True
                                  )

    print(len(data_loader))
    for image,label in data_loader:
        print("image",image.shape)
        # label = torch.tensor(label)
        print("labels:",label)













































































































































