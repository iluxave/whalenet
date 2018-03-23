import math
from PyQt4.Qt import *
import sloth
from sloth.items import PolygonItem
from sloth.plugins.facedetector import FaceDetectorPlugin
from sloth.annotations.model import ImageModelItem

import tensorflow as tf

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

import numpy as np
import cv2

from os.path import dirname

BASEDIR=dirname(__file__)

PATH_TO_CKPT="/home/ilya/ai/tf-models/research/object_detection/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb"
PATH_TO_LABELS="/home/ilya/ai/tf-models/research/object_detection/data/mscoco_label_map.pbtxt"
NUM_CLASSES = 90

class EditablePolygonItem(PolygonItem):
    def __init__(self, *args, **kwargs):
        sloth.items.PolygonItem.__init__(self, *args, **kwargs)

        self.point_in_motion = None
        self.point_translate_start = None

    def find_nearest_point(self, x, y, max_dist = 10.0):
        dists = [math.sqrt((x - pt.x()) ** 2 + (y - pt.y()) ** 2) for pt in self._polygon]
        dist, min_ind = min((val, idx) for (idx, val) in enumerate(dists))
        if max_dist is not None and max_dist < dist:
            return None
        return min_ind

    def mousePressEvent(self, event):
        min_ind = self.find_nearest_point(event.pos().x(), event.pos().y())
        if event.modifiers() == Qt.ControlModifier:
            if min_ind is not None:
                self._polygon.remove(min_ind)
        elif event.modifiers() == Qt.ShiftModifier:
            #min_ind = self.find_nearest_point(event.pos().x(), event.pos().y(), None)
            self.create_new_point(event.pos())
        else:
            if event.button() == Qt.LeftButton:
                if min_ind is not None:
                    self.point_in_motion = min_ind
                else:
                    self.point_translate_start = event.pos()
            else:
                PolygonItem.mousePressEvent(self, event)
        event.accept()


    def mouseMoveEvent(self, event):
        poly = self._polygon
        if self.point_in_motion is not None:
            self.prepareGeometryChange()
            poly[self.point_in_motion] = event.pos()
        elif self.point_translate_start is not None:
            self.prepareGeometryChange()
            poly.translate(event.pos().x()-self.point_translate_start.x(),
                           event.pos().y()-self.point_translate_start.y())
            self.point_translate_start = event.pos()
        else:
            PolygonItem.mouseMoveEvent(self, event)
        event.accept()

    def mouseReleaseEvent(self, event):
        self.point_in_motion = None
        self.point_translate_start = None
        self.updateModel()
        event.accept()

    def create_new_point(self, pt):
        ptind = self.find_nearest_point(pt.x(), pt.y(), None)
        if ptind is None:
            return
        if ptind == 0:
            ptprev = self._polygon.size()
        else:
            ptprev = ptind - 1
        if ptind == self._polygon.size()-1:
            ptnext = 0
        else:
            ptnext = ptind + 1
        ptp = self._polygon[ptprev]
        ptn = self._polygon[ptnext]
        distprev = (pt.x() - ptp.x())**2+(pt.y()-ptp.y())**2
        distnext = (pt.x() - ptn.x())**2+(pt.y()-ptn.y())**2
        if(distprev < distnext):
            self._polygon.insert(ptind, pt)
        else:
            self._polygon.insert(ptind+1, pt)
        self.updateModel()

    def updateModel(self):
        xns = [str(pt.x()) for pt in self._polygon]
        yns = [str(pt.y()) for pt in self._polygon]
        xn_string = ";".join(xns)
        yn_string = ";".join(yns)
        self._model_item.update({
            self.prefix() + 'xn': xn_string,
            self.prefix() + 'yn': yn_string,
        })

    def paint(self, painter, option, widget=None):
        sloth.items.PolygonItem.paint(self, painter, option, widget)
        for pt in self._polygon:
            painter.drawEllipse(QRectF(pt.x()-4, pt.y()-4, 8, 8))

    def boundingRect(self):
        rect = PolygonItem.boundingRect(self)
        xp1, yp1, xp2, yp2 = rect.getCoords()
        rect.setCoords(xp1-4, yp1-4, xp2+4, yp2+4)
        return rect

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



LABELS = (
    {
        'attributes': {
            'class':    'tail',
        },
        'inserter': 'sloth.items.PolygonItemInserter',
        'item':     EditablePolygonItem,
        'hotkey':   't',
        'text':     'Tail',
    },
    {
        'attributes': {
            'class':    'left_fluke',
        },
        'inserter': 'sloth.items.PolygonItemInserter',
        'item':     EditablePolygonItem,
        'hotkey':   'l',
        'text':     'Left Fluke',
    },
    {
        'attributes': {
            'class':    'right_fluke',
        },
        'inserter': 'sloth.items.PolygonItemInserter',
        'item':     EditablePolygonItem,
        'hotkey':   'r',
        'text':     'Right Fluke',
    },
)

PLUGINS = [
    WhaleDetectorPlugin
]
