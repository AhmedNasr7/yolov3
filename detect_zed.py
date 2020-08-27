import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from utils.zed_utils import *



class ZED_Detector: 



    def __init__(self, opt, source_img, cfg, weights, save_img=False):
        #soource_img, weilghts_path, 
        #self.source_img = source_img
        self.img_size = 416
        self.cfg = cfg
        self.device = torch_utils.select_device(device='0')
        self.weights = weights
        #self.save_img = save_img

        self.model = Darknet(self.cfg, self.img_size)   # Initialize model
        self.opt = opt
        
        # Load weights
        #attempt_download(self.weights) 
        
        if self.weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(self.weights, map_location=self.device)['model']) ## model loading is here
        else:  # darknet format
            load_darknet_weights(self.model, self.weights) ## or here???
        
        self.model.to(self.device).eval() ## evaluation mode


        self.half = True

        # Half precision 
        #self.half = self.half and self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model.half()

        self.agnostic_nms =  ### ?

        # Get names and colors
        self.names = '/content/robosub.names'
        self.names = load_classes(self.names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.classes = self.opt.classes

        self.half = self.half and self.device.type != 'cpu'  # half precision only supported on CUDA

        #self.dataset = LoadImages(self.source_img, img_size=self.img_size, half=self.half)



        self.detections= []
        self.confidence = -1 # initially


    def init_ZED(self):

        self.zed = sl.Camera()

        # Set configuration parameters
        input_type = sl.InputType()
        if len(sys.argv) >= 2 :
            input_type.set_from_svo_file(sys.argv[1])
        init = sl.InitParameters(input_t=input_type)
        init.camera_resolution = sl.RESOLUTION.HD1080
        init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        init.coordinate_units = sl.UNIT.MILLIMETER

        # Open the camera
        self.err = zed.open(init)
        if self.err != sl.ERROR_CODE.SUCCESS :
            print(repr(self.err))
            self.zed.close()
            print("Cant open ZED Camera!")
            exit(1)

        # Set runtime parameters after opening the camera
        runtime = sl.RuntimeParameters()
        runtime.sensing_mode = sl.SENSING_MODE.STANDARD

        # Prepare new image size to retrieve half-resolution images
        image_size = self.zed.get_camera_information().camera_resolution
        image_size.width = image_size.width /2
        image_size.height = image_size.height /2

        # Declare your sl.Mat matrices
        self.image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
        #depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
        #point_cloud = sl.Mat()


    def detect(self, conf_thres = 0.3, iou_thres = 0.5):


        self.err = zed.grab(runtime)
        if self.err == sl.ERROR_CODE.SUCCESS :
            # Retrieve the left image, depth image in the half-resolution
            self.zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            self.zed.retrieve_image(depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)
            # Retrieve the RGBA point cloud in half resolution
            self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, image_size)

            # To recover data from sl.Mat to use it with opencv, use the get_data() method
            # It returns a numpy array that can be used as a matrix with opencv
            self.image_np = self.image_zed.get_data()
            #depth_image_ocv = depth_image_zed.get_data()

            #cv2.imshow("Image", image_ocv)
            #cv2.imshow("Depth", depth_image_ocv)

            #key = cv2.waitKey(10)

            #process_key_event(zed, key)

            #zed.close()



            self.conf_thres = conf_thres
            self.iou_thres = iou_thres
      

            # Get detections
            img = torch.from_numpy(self.image_np).to(self.device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = self.model(img)[0]
            if self.half:
                pred = pred.float()

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.opt.classes, agnostic=self.agnostic_nms)

          
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                               
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in det:

                        detection = [int(cls.item()), xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]
                        self.detections.append(detection)
                        self.confidence = conf.item()

                        if self.save_img:  # Add bbox to image
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)])

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, time.time() - t))

                # Save results (image with detections)
                if self.save_img:
                    if self.dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)
                    
        return self.detections, self.confidence

    def __del__(self): 
        self.zed.close()
        print('Destructor called, Camera Closed.') 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    source_img = '/content/1.jpg'
    cfg = '/content/yolov3-spp.cfg'
    weights = './best-AUV-gate.weights'
    opt = parser.parse_args()
    detector = Detector(opt, source_img, cfg, weights, save_img=True)
    detections, confidence = detector.detect(0.5, 0.1)
    print(detections)
    print("confidence: ", confidence)