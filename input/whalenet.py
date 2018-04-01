from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
import cv2
from PyQt5.Qt import *
import tensorflow as tf
import numpy as np

from sloth.annotations.model import ImageModelItem

import object_detection
import os
objdet_path=os.path.dirname(object_detection.__file__)


PATH_TO_CKPT=os.path.join(objdet_path, "mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28","frozen_inference_graph.pb")
PATH_TO_LABELS=os.path.join(objdet_path, "data", "mscoco_label_map.pbtxt")
NUM_CLASSES = 90

# This class runs the image through a google object detection API (using mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28)
# And generates polygons based on detected masks
class WhaleDetectorPlugin(QObject):
    def __init__(self, labeltool):
        QObject.__init__(self)
        self._labeltool = labeltool
        self._wnd = labeltool.mainWindow()
        self._sc  = QAction("Detect whales", self._wnd)
        self._sc.triggered.connect(self.doit)
        # Prepare the TF model and stuff
        self._graph = tf.Graph()
        with self._graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        self._categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self._category_index = label_map_util.create_category_index(self._categories)

    # Return QPolygonF with a polygon bounding the mask. The polygon might
    # be smoothed out a bit.
    def mask2poly(self, mask):
        mask2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        poly=QPolygonF()
        for contour in contours:
            for pt in contour:
                poly.append(QPointF(pt[0][0], pt[0][1]))
        return poly

    def run_inference_for_single_image(self, image):
        if len(image.shape) == 2:
            image=np.stack((image,)*3, -1)
        with self._graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                              feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                        'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def doit(self):
        self._sc.setEnabled(False)
        model = self._labeltool.model()
        n_images = model.rowCount()
        for i, item in enumerate(model.iterator(ImageModelItem)):
            img = self._labeltool.getImage(item)
            whales = self.run_inference_for_single_image(img)
            for whale in whales['detection_masks']:
                whale_poly=self.mask2poly(whale)
                ann = {
                        'class':    'tail',
                        'xn':        ';'.join([str(p.x()) for p in whale_poly]),
                        'yn':        ';'.join([str(p.y()) for p in whale_poly]),
                        'autodetected': 'true',
                        'corrected': 'false',
                      }

                item.addAnnotation(ann)
        self._sc.setEnabled(True)

    def action(self):
        return self._sc
