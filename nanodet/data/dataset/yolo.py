# Copyright 2023 cansik.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import time
from collections import defaultdict
from typing import Optional, Sequence
import threading
import json

import numpy as np
from imagesize import imagesize
from pycocotools.coco import COCO

from .coco import CocoDataset
from .xml_dataset import get_file_list


def yolo_worker_coco(yolodataset ,worker, chunk_size,class_names, ann_path, ann_file_names):
    
    image_info, categories, annotations = YoloDataset.yolo_worker_coco(worker, chunk_size,class_names, ann_path, ann_file_names)
    yolodataset.lock.acquire()
    yolodataset.image_info = yolodataset.image_info + image_info
    yolodataset.categories = yolodataset.categories + categories
    yolodataset.annotations = yolodataset.annotations + annotations
    yolodataset.lock.release()
    
    

class CocoYolo(COCO):
    def __init__(self, annotation):
        """
        Constructor of Microsoft COCO helper class for
        reading and visualizing annotations.
        :param annotation: annotation dict
        :return:
        """
        # load dataset
        super().__init__()
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        dataset = annotation
        assert type(dataset) == dict, "annotation file format {} not supported".format(
            type(dataset)
        )
        self.dataset = dataset
        self.createIndex()


class YoloDataset(CocoDataset):
    def __init__(self, class_names, **kwargs):
        self.class_names = class_names
        super(YoloDataset, self).__init__(**kwargs)
        self.image_info = []
        self.categories = []
        self.annotations = []
        self.lock = threading.Lock()
    # @staticmethod
    # def _find_image(
    #     image_prefix: str,
    #     image_types: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff"),
    # ) -> Optional[str]:
    #     for image_type in image_types:
    #         path = f"{image_prefix}{image_type}"
    #         if os.path.exists(path):
    #             return path
    #     return None

    @staticmethod
    def _find_image(
        image_prefix: str,
        image_types: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff"),
    ) -> Optional[str]:
        """
        Tìm kiếm tệp hình ảnh dựa trên tên tệp annotation không có phần mở rộng `.txt`.

        :param image_prefix: Đường dẫn tệp không có phần mở rộng `.txt`.
        :param image_types: Danh sách các phần mở rộng ảnh hợp lệ.
        :return: Đường dẫn tệp hình ảnh nếu tìm thấy, ngược lại trả về None.
        """
        for image_type in image_types:
            path = f"{image_prefix}{image_type}"
            if os.path.exists(path):
                return path
        return None
    
    @staticmethod
    def yolo_worker_coco(worker, chunk_size,class_names, ann_path, ann_file_names):
        logging.info("loading annotations into memory worker %d  ..." %worker)
        tic = time.time()
        logging.info("Found {} annotation files.".format(len(ann_file_names)))
        image_info = []
        categories = []
        annotations = []
        for idx, supercat in enumerate(class_names):
            categories.append(
                {"supercategory": supercat, "id": idx + 1, "name": supercat}
            )
            
        ann_id = 1 + worker* chunk_size
        
        for idx, txt_name in enumerate(ann_file_names):
            ann_file = os.path.join(ann_path, txt_name)
            # print(os.path.splitext(ann_file)[0])
            image_file = YoloDataset._find_image(os.path.splitext(ann_file)[0])
            # print(image_file)
            if image_file is None:
                logging.warning(f"Could not find image for {ann_file}")
                continue

            with open(ann_file, "r") as f:
                lines = f.readlines()

            width, height = imagesize.get(image_file)

            file_name = os.path.basename(image_file)
            info = {
                "file_name": file_name,
                "height": height,
                "width": width,
                "id": idx + 1,
            }
            image_info.append(info)
            for line in lines:
                data = [float(t) for t in line.split(" ")]
                cat_id = int(data[0])
                locations = np.array(data[1:]).reshape((len(data) // 2, 2))
                bbox = locations[0:2]

                bbox[0] -= bbox[1] * 0.5

                bbox = np.round(bbox * np.array([width, height])).astype(int)
                x, y = bbox[0][0], bbox[0][1]
                w, h = bbox[1][0], bbox[1][1]

                if cat_id >= len(class_names):
                    logging.warning(
                        f"Category {cat_id} is not defined in config ({txt_name})"
                    )
                    continue

                if w < 0 or h < 0:
                    logging.warning(
                        "WARNING! Find error data in file {}! Box w and "
                        "h should > 0. Pass this box annotation.".format(txt_name)
                    )
                    continue

                coco_box = [max(x, 0), max(y, 0), min(w, width), min(h, height)]
                ann = {
                    "image_id": idx + 1,
                    "bbox": coco_box,
                    "category_id": cat_id + 1,
                    "iscrowd": 0,
                    "id": ann_id,
                    "area": coco_box[2] * coco_box[3],
                }
                annotations.append(ann)
                ann_id += 1

        # coco_dict = {
        #     "images": image_info,
        #     "categories": categories,
        #     "annotations": annotations,
        # }
        # logging.info(
        #     "Load {} txt files and {} boxes".format(len(image_info), len(annotations))
        # )
        # logging.info("Done (t={:0.2f}s)".format(time.time() - tic))
        return image_info, categories, annotations
    
    def split_list(self, lst, worker):
        chunk_size = int (len(lst) / worker)
        
        print("chunk_size ==========================: ", chunk_size)
        
        if (chunk_size <=0):
            return 1, [lst]
        
        return chunk_size , [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def save_coco_list(self, data, ann_path):
        # /media/tamnln/DATA/nanodet/train/coco-list.json
        file = '%s/../coco-list.json' % ann_path
        f = open(file, "a")
        f.write(json.dumps(data,indent=2))
        f.close()
        
    def load_coco_file(self, ann_path):
        file = '%s/../coco-list.json' % ann_path
        if (os.path.isfile(file)):

            f = open(file, "r")
            data = f.read()
            f.close()
            return json.loads(data)
        return None
        

    def yolo_to_coco(self, ann_path):
        """
        convert yolo annotations to coco_api
        :param ann_path:
        :return:
        """
        print ("ann_path: ", ann_path)
        logging.info("loading annotations into memory...")
        tic = time.time()
        ann_file_names = get_file_list(ann_path, type=".txt")
        logging.info("Found {} annotation files.".format(len(ann_file_names)))
        
        data = self.load_coco_file(ann_path)
        if data :
            return data
        
        # data = {'haha':'hehe'}
        # self.save_coco_list(data, ann_path)
        
        # worker = 512
        
        # chunk_size ,ann_list = self.split_list(ann_file_names, worker)
        
        # image_info = []
        # categories = []
        # annotations = []
        # thread_list = []
        # for i in range(0, worker):
        #     thread = threading.Thread(target = yolo_worker_coco, args = (self, worker, chunk_size, self.class_names, ann_path, ann_file_names, ))
        #     thread_list.append(thread)
        #     thread.start()
        
        # for i in range(0, worker):
        #     thread.join()
        
        # coco_dict = {
        #     "images": self.image_info,
        #     "categories": self.categories,
        #     "annotations": self.annotations,
        # }
        # logging.info(
        #     "Load {} txt files and {} boxes".format(len(image_info), len(annotations))
        # )
        # logging.info("Done (t={:0.2f}s)".format(time.time() - tic))
        # return coco_dict
        
        # return 
        image_info = []
        categories = []
        annotations = []
        for idx, supercat in enumerate(self.class_names):
            categories.append(
                {"supercategory": supercat, "id": idx + 1, "name": supercat}
            )
        ann_id = 1

        for idx, txt_name in enumerate(ann_file_names):
            ann_file = os.path.join(ann_path, txt_name)
            # print(os.path.splitext(ann_file)[0])
            image_file = self._find_image(os.path.splitext(ann_file)[0])
            print(image_file)
            if image_file is None:
                logging.warning(f"Could not find image for {ann_file}")
                continue

            with open(ann_file, "r") as f:
                lines = f.readlines()

            width, height = imagesize.get(image_file)

            file_name = os.path.basename(image_file)
            info = {
                "file_name": file_name,
                "height": int(height),
                "width": int(width),
                "id": int(idx + 1),
            }
            image_info.append(info)
            for line in lines:
                data = [float(t) for t in line.split(" ")]
                cat_id = int(data[0])
                locations = np.array(data[1:]).reshape((len(data) // 2, 2))
                bbox = locations[0:2]

                bbox[0] -= bbox[1] * 0.5

                bbox = np.round(bbox * np.array([width, height])).astype(int)
                x, y = bbox[0][0], bbox[0][1]
                w, h = bbox[1][0], bbox[1][1]

                if cat_id >= len(self.class_names):
                    logging.warning(
                        f"Category {cat_id} is not defined in config ({txt_name})"
                    )
                    continue

                if w < 0 or h < 0:
                    logging.warning(
                        "WARNING! Find error data in file {}! Box w and "
                        "h should > 0. Pass this box annotation.".format(txt_name)
                    )
                    continue

                coco_box = [int(max(x, 0)), int(max(y, 0)), int(min(w, width)), int(min(h, height))]
                ann = {
                    "image_id": int(idx + 1),
                    "bbox": coco_box,
                    "category_id": cat_id + 1,
                    "iscrowd": 0,
                    "id": int(ann_id),
                    "area": int(coco_box[2] * coco_box[3]),
                }
                annotations.append(ann)
                ann_id += 1

        coco_dict = {
            "images": image_info,
            "categories": categories,
            "annotations": annotations,
        }
        logging.info(
            "Load {} txt files and {} boxes".format(len(image_info), len(annotations))
        )
        logging.info("Done (t={:0.2f}s)".format(time.time() - tic))
        
        self.save_coco_list(coco_dict, ann_path)
        return coco_dict

    def get_data_info(self, ann_path):
        """
        Load basic information of dataset such as image path, label and so on.
        :param ann_path: coco json file path
        :return: image info:
        [{'file_name': '000000000139.jpg',
          'height': 426,
          'width': 640,
          'id': 139},
         ...
        ]
        """
        coco_dict = self.yolo_to_coco(ann_path)
        self.coco_api = CocoYolo(coco_dict)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)
        return img_info
