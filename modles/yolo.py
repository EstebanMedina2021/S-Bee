import colorsys
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image, show_config)
from utils.utils_bbox import decode_outputs, non_max_suppression

class YOLO(object):
    _defaults = {
        "model_path"        : 'logs/best_epoch_weights.pth',
        "classes_path"      : 'model_data/voc_classes.txt',
        "input_shape"       : [640, 640],
        "phi"               : 's',
        "confidence"        : 0.7,
        "nms_iou"           : 0.3,
        "letterbox_image"   : True,
        "cuda"              : False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 

        self.class_names, self.num_classes  = get_classes(self.classes_path)

        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()
        
        show_config(**self._defaults)

    def generate(self, onnx=False):
        device = torch.device('cuda' if self.cuda and torch.cuda.is_available() else 'cpu')  # Cambio realizado aquí
        self.net = YoloBody(self.num_classes, self.phi)
        self.net.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image(self, image, supervision=False):
        #---------------------------------------------------#
        # Get the height and width of the input image
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])

        #---------------------------------------------------------#
        # Convert the image to an RGB image here to prevent greyscale maps from reporting errors in the prediction.
        # The code only supports prediction of RGB images, all other types of images are converted to RGB.
        #---------------------------------------------------------#
        image = cvtColor(image)

        #---------------------------------------------------------#
        # Add grey bars to the image to achieve undistorted resize
        # Can also resize directly for recognition
        #---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        #---------------------------------------------------------#
        # Add the batch_size dimension.
        #---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        font_path = 'model_data/Roboto-Black.ttf'
        font_size = np.floor(3e-2 * image.size[1] + 0.5).astype('int32')
        class_name_to_id = {
            'varroa': 0,
            'Bee': 1
        }
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            
            #---------------------------------------------------------#
            # Feed the image into the network for prediction!
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)

            #---------------------------------------------------------#
            # Stack the prediction frames and then perform non-great suppression
            #---------------------------------------------------------#
            results = non_max_suppression(outputs, 
                                          self.num_classes, 
                                          self.input_shape, 
                                          image_shape, 
                                          self.letterbox_image, 
                                          conf_thres=self.confidence, 
                                          nms_thres=self.nms_iou)

            if results[0] is None:
                return image

        top_label = np.array(results[0][:, 6], dtype='int32')
        top_conf = results[0][:, 4] * results[0][:, 5]
        top_boxes = results[0][:, :4]
        # Convert predicted class names to numeric IDs
        predicted_classes = [class_name_to_id[self.class_names[int(Bee)]] for i, Bee in enumerate(top_label)]

        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        for i, Bee in list(enumerate(top_label)):
            predicted_class = self.class_names[int(Bee)]
            box = top_boxes[i]
            score = top_conf[i] 
            top, left, bottom, right = box
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)

            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(font_path, font_size)
            # Get text width using getlength
            label_size = font.getlength(label)

            # Estimate text height (assuming similar to width)
            estimated_height = label_size

            label = label.encode('utf-8')

            if top - estimated_height >= 0:
                text_origin = np.array([left, top - estimated_height])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[Bee])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[Bee])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw


        if not supervision:
            return image
        else:
            if len(top_conf) > 0:
                detections = [
                    box,
                    np.array([top_conf[0]]),
                    np.array([predicted_classes[0]]),
                ]
            else:
                # Handle the case when no detections are found
                detections = []
            return image, detections

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names   = ["images"]
        output_layer_names  = ["output"]
        
        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                        im,
                        f               = model_path,
                        verbose         = False,
                        opset_version   = 12,
                        training        = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding = True,
                        input_names     = input_layer_names,
                        output_names    = output_layer_names,
                        dynamic_axes    = None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))
        