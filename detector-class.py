
import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import cv2
import time


class Detector:

    def __init__(self, source_img, cfg, weights, save_img=False):
        #soource_img, weilghts_path, 
        self.source_img = source_img
        img_size = 416 ## to be changed
        self.cfg = cfg
        self.device = torch_utils.select_device(device='0')
        self.weights = weights

        self.model = Darknet(self.cfg, img_size)   # Initialize model
        
        # Load weights
        attempt_download(self.weights) 
   		
        if self.weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(self.weights, map_location=self.device)['model']) ## model loading is here
        else:  # darknet format
            load_darknet_weights(self.model, self.weights) ## or here???
        
        self.model.to(self.device).eval() ## evaluation mode

        if save_img:
            save_path = './' # to the current dir

        self.half = True

        # Half precision 

        self.half = self.half and self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model.half()

        self.classes = ''
        self.agnostic_nms = True

        # Get names and colors
        self.names = './data/coco.names'
        self.names = load_classes(self.names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]


    def detect(self, conf_thres = 0.3, iou_thres = 0.5):


        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
  

        img = cv2.imread(self.source_img) ## to be changed

        #for path, img, im0s, vid_cap in self.dataset:
        #t = time.time()

        # Run inference
        t0 = time.time()

        img = img.reshape(img.shape[0], img.shape[1], 3)

        img = img.transpose((2, 0, 1))

        img = torch.from_numpy(img).to(self.device)

        with torch.no_grad():

        	pred = self.model(img.unsqueeze())[0]

	        if self.half:
	            pred = pred.float()

	        # Apply NMS
	        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

	       
	        # Process detections
	        for i, det in enumerate(pred):  # detections per image
	           
	            save_path = str(Path(self.out) / Path(p).name)
	            s += '%gx%g ' % img.shape[2:]  # print string
	            
	            if det is not None and len(det):
	                # Rescale boxes from img_size to im0 size
	                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
	                print('detection.shape : ',det.shape)
	                print('detection : ',det)
	                # Print results
	                for c in det[:, -1].unique():
	                    n = (det[:, -1] == c).sum()  # detections per class
	                    s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

	                # Write results
	                for *xyxy, conf, cls in det:
	                    if self.save_img or self.view_img:  # Add bbox to image
	                        label = '%s %.2f' % (self.names[int(cls)], conf)
	                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)])

	            # Print time (inference + NMS)
	            print('%sDone. (%.3fs)' % (s, time.time() - t0))


	            # Save results (image with detections)
	            if self.save_img:
	            	cv2.imwrite(save_path, im0)
	               
	        if self.save_txt or self.save_img:
	            print('Results saved to %s' % os.getcwd() + os.sep + self.out)
	            if platform == 'darwin':  # MacOS
	                os.system('open ' + self.out + ' ' + save_path)

	        print('Done. (%.3fs)' % (time.time() - t0))



if __name__ == '__main__':
	source_img = './img.jpg'
	cfg = './cfg/yolov3-spp.cfg'
	weights = './weights/yolov3-spp.weights'

	detector = Detector(source_img, cfg, weights, save_img=True)
	detector.detect()

