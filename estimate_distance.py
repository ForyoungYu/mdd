'''
Create July 27, 2020
@author ClearTorch
'''

import pandas as pd
import numpy as np
import sys
import platform
import math
import csv
import argparse
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

# from models.experimental import attempt_load
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
# from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
#     scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
# from utils.plotting import colors, plot_one_box
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh, set_logging)
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from utils.torch_utils import select_device, reshape_classifier_output, time_sync
from utilsd import convert_arg_line_to_args

import plotly.graph_objects as go   #Plotly是一个基于Javascript的绘图库,绘图工具一般是graph_objects工具

start_time = time.time()
print('Pandas Version:', pd.__version__)
print('Nunpy Version:', np.__version__)

@torch.no_grad()

class DistanceEstimation:
    def __init__(self):
        self.W = 640
        self.H = 480
        self.excel_path = r'./camera_parameters.xlsx'

    def camera_parameters(self, excel_path):
        df_intrinsic = pd.read_excel(excel_path, sheet_name='内参矩阵', header=None)
        df_p = pd.read_excel(excel_path, sheet_name='外参矩阵', header=None)

        # print('外参矩阵形状：', df_p.values.shape)
        # print('内参矩阵形状：', df_intrinsic.values.shape)

        return df_p.values, df_intrinsic.values

    def object_point_world_position(self, u, v, w, h, p, k):
        u1 = u
        v1 = v + h / 2
        # print('关键点坐标：', u1, v1)

        alpha = -(90 + 0) / (2 * math.pi)
        peta = 0
        gama = -90 / (2 * math.pi)

        fx = k[0, 0]
        fy = k[1, 1]
        H = 1
        angle_a = 0
        angle_b = math.atan((v1 - self.H / 2) / fy)
        angle_c = angle_b + angle_a
        # print('angle_b', angle_b)

        depth = (H / np.sin(angle_c)) * math.cos(angle_b)
        print('depth', depth)

        k_inv = np.linalg.inv(k)
        p_inv = np.linalg.inv(p)
        # print(p_inv)
        point_c = np.array([u1, v1, 1])
        point_c = np.transpose(point_c)
        # print('point_c', point_c)
        # print('k_inv', k_inv)
        c_position = np.matmul(k_inv, depth * point_c)
        # print('c_position', c_position)
        c_position = np.append(c_position, 1)
        c_position = np.transpose(c_position)
        c_position = np.matmul(p_inv, c_position)
        d1 = np.array((c_position[0], c_position[1]), dtype=float)
        return d1


    def distance(self, box, xw=5, yw=0.1):
        # print('=' * 50)
        # print('开始测距')
        fig = go.Figure()
        p, k = self.camera_parameters(self.excel_path)
        if len(box):
            obj_position = []
            u, v, w, h = box[1] * self.W, box[2] * self.H, box[3] * self.W, box[4] * self.H
            # print('目标框', u, v, w, h)
            d1 = self.object_point_world_position(u, v, w, h, p, k)
        distance = 0
        print('距离', d1)
        if d1[0] <= 0:
            d1[:] = 0
        else:
            distance = math.sqrt(math.pow(d1[0], 2) + math.pow(d1[1], 2))
        print("distance:", distance)
        print("d1:", d1)
        return distance, d1


    def Detect(self, weights='yolov5s.pt',
           source='data/images',  # file/dir/URL/glob, 0 for webcam
           data='data/coco128.yaml',  # data.yaml path
           imgsz=(640, 640),  # inference size (height, width)
           conf_thres=0.25,  # confidence threshold
           iou_thres=0.45,  # NMS IOU threshold
           max_det=1000,  # maximum detections per image
           device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
           view_img=False,  # show results
           save_txt=False,  # save results to *.txt
           save_csv=False,  # save results in CSV format
           save_conf=False,  # save confidences in --save-txt labels
           save_crop=False,  # save cropped prediction boxes
           nosave=False,  # do not save images/videos
           classes=None,  # filter by class: --class 0, or --class 0 2 3
           threshold=10.0,  # distence threshold
           agnostic_nms=False,  # class-agnostic NMS
           augment=False,  # augmented inference
           visualize=False,  # visualize features
           update=False,  # update all models
           project='runs/detect',  # save results to project/name
           name='exp',  # save results to project/name
           exist_ok=False,  # existing project/name ok, do not increment
           line_thickness=3,  # bounding box thickness (pixels)
           hide_labels=False,  # hide labels
           hide_conf=False,  # hide confidences
           half=False,  # use FP16 half-precision inference
           dnn=False
           ):
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download
        
        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size  测距不要缩放图片

        # Dataloader
        bs = 1
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
        # vid_path, vid_writer = None, None
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Define the path for the CSV file
            csv_path = save_dir / 'predictions.csv'

            # Create or append to the CSV file
            def write_to_csv(image_name, prediction, confidence):
                data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
                with open(csv_path, mode='a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=data.keys())
                    if not csv_path.is_file():
                        writer.writeheader()
                    writer.writerow(data)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = names[c] if hide_conf else f'{names[c]}'
                        confidence = float(conf)
                        confidence_str = f'{confidence:.2f}'
                        
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        box = [int(cls), xywh[0], xywh[1], xywh[2], xywh[3]]
                        
                        if save_csv:
                            write_to_csv(p.name, label, confidence_str)
                        
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
    
                        distance, d = self.distance(box)
    
                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            if label != None and distance != 0:
                                if d[0] > threshold:
                                    # label = label + ' ' + str('%.1f' % d[0]) + 'm'+ str('%.1f' % d[1]) + 'm'
                                    label = label + ' ' + str('%.1f' % d[0]) + 'm'
                                else: # icon: 
                                    label = label + ' !' + str('%.1f' % d[0]) + 'm'
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond
    
                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PixelFormer PyTorch implementation.', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save_txt',default=False, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--threshold', type=float, default=10.0, help='distence threshold') 
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        opt = parser.parse_args([arg_filename_with_prefix])
    else:
        opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    
    print_args(vars(opt))
    # check_requirements(exclude=('tensorboard', 'thop'))

    print('开始进行目标检测和单目测距！')
    DE = DistanceEstimation()
    DE.Detect(**vars(opt))
    if time.time()>(start_time + 10):
        cv2.waitKey (0)
        cv2.destroyAllWindows()









