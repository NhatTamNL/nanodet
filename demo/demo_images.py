import argparse
import os
import time
import cv2
import torch

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.util.path import mkdir

image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("demo", default="image", help="demo type, eg. image")
    parser.add_argument("--config", help="model config file path")
    parser.add_argument("--model", help="model file path")
    parser.add_argument("--path", default="./demo", help="path to images")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    args = parser.parse_args()
    return args

class Predictor(object):
    def __init__(self, cfg, model_path, logger, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    # def inference(self, img):
    #     img_info = {"id": 0}
    #     if isinstance(img, str):
    #         img_info["file_name"] = os.path.basename(img)
    #         img = cv2.imread(img)
    #     else:
    #         img_info["file_name"] = None

    #     height, width = img.shape[:2]
    #     img_info["height"] = height
    #     img_info["width"] = width
    #     meta = dict(img_info=img_info, raw_img=img, img=img)
    #     meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
    #     meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
    #     meta = naive_collate([meta])
    #     meta["img"] = stack_batch_img(meta["img"], divisible=32)
    #     with torch.no_grad():
    #         results = self.model.inference(meta)
    #     return meta, results
    
    def inference(self, img):
        img_info = {"id": 0}
        
        # Kiểm tra nếu 'img' là chuỗi đường dẫn, thì đọc ảnh
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
            
            # Kiểm tra xem ảnh có được đọc thành công không
            if img is None:
                print(f"Warning: Failed to read image {img_info['file_name']}. Skipping...")
                return None, None
        else:
            img_info["file_name"] = None
    
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        
        with torch.no_grad():
            results = self.model.inference(meta)
        
        return meta, results


    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        result_img = self.model.head.show_result(
            meta["raw_img"][0], dets, class_names, score_thres=score_thres, show=True
        )
        return result_img

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names

def main():
    args = parse_args()
    local_rank = 0
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    load_config(cfg, args.config)
    logger = Logger(local_rank, use_tensorboard=False)
    predictor = Predictor(cfg, args.model, logger, device="cuda:0")
    
    logger.log('Press "Esc", "q" or "Q" to exit.')
    current_time = time.localtime()
    
    if args.demo == "image":
        if os.path.isdir(args.path):
            files = get_image_list(args.path)
        else:
            files = [args.path]
        
        files.sort()
        save_folder = os.path.join(
            cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        mkdir(local_rank, save_folder)
        
        for image_name in files:
            meta, res = predictor.inference(image_name)

            # Bỏ qua nếu inference không trả về kết quả (ảnh không đọc được)
            if meta is None or res is None:
                continue
            
            score_threshold = 0.5  # Ngưỡng điểm

            if isinstance(res[0], list):
                detected_objects = [det for det in res[0] if len(det) > 0 and det[-1] >= score_threshold]

                if detected_objects:
                    result_image = predictor.visualize(detected_objects, meta, cfg.class_names, score_threshold)
                    if args.save_result:
                        save_file_name = os.path.join(save_folder, os.path.basename(image_name))
                        cv2.imwrite(save_file_name, result_image)

            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break


if __name__ == "__main__":
    main()
