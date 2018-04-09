import math
from PyQt5.Qt import *
import sloth
from sloth.items import PolygonItem
import numpy as np

whaledetect=0
import traceback
try:
    from object_detection.utils import ops as utils_ops
    from object_detection.utils import label_map_util
    from whalenet import WhaleDetectorPlugin
    whaledetect=1
    print("Whale detection enabled")
except Exception as e:
    print("Whale detection disabled")
    traceback.print_exc()
    pass

from os.path import dirname

BASEDIR=dirname(__file__)

def point_to_seg_dist(point, line):
    """Calculate the distance between a point and a line segment.

    To calculate the closest distance to a line segment, we first need to check
    if the point projects onto the line segment.  If it does, then we calculate
    the orthogonal distance from the point to the line.
    If the point does not project to the line segment, we calculate the 
    distance to both endpoints and take the shortest distance.

    :param point: Numpy array of form [x,y], describing the point.
    :type point: numpy.core.multiarray.ndarray
    :param line: list of endpoint arrays of form [P1, P2]
    :type line: list of numpy.core.multiarray.ndarray
    :return: The minimum distance to a point.
    :rtype: float

    Source: https://stackoverflow.com/a/45483585/7724174
    """
    # unit vector
    unit_line = line[1] - line[0]
    norm_unit_line = unit_line / np.linalg.norm(unit_line)

    # compute the perpendicular distance to the theoretical infinite line
    segment_dist = (
        np.linalg.norm(np.cross(line[1] - line[0], line[0] - point)) /
        np.linalg.norm(unit_line)
    )

    diff = (
        (norm_unit_line[0] * (point[0] - line[0][0])) + 
        (norm_unit_line[1] * (point[1] - line[0][1]))
    )

    x_seg = (norm_unit_line[0] * diff) + line[0][0]
    y_seg = (norm_unit_line[1] * diff) + line[0][1]

    endpoint_dist = min(
        np.linalg.norm(line[0] - point),
        np.linalg.norm(line[1] - point)
    )

    # decide if the intersection point falls on the line segment
    lp1_x = line[0][0]  # line point 1 x
    lp1_y = line[0][1]  # line point 1 y
    lp2_x = line[1][0]  # line point 2 x
    lp2_y = line[1][1]  # line point 2 y
    is_betw_x = lp1_x <= x_seg <= lp2_x or lp2_x <= x_seg <= lp1_x
    is_betw_y = lp1_y <= y_seg <= lp2_y or lp2_y <= y_seg <= lp1_y
    if is_betw_x and is_betw_y:
        return segment_dist
    else:
        # if not, then return the minimum distance to the segment endpoints
        return endpoint_dist

class EditablePolygonItem(PolygonItem):
    def __init__(self, model_item=None, prefix="", parent=None):
        sloth.items.PolygonItem.__init__(self, model_item, prefix, parent)

        self.point_in_motion = None
        self.point_translate_start = None
        if 'corrected' in model_item:
            self._corrected = model_item['corrected']
        else:
            self._corrected = False


    def find_nearest_point(self, x, y, max_dist = 10.0):
        dists = [math.sqrt((x - pt.x()) ** 2 + (y - pt.y()) ** 2) for pt in self._polygon]
        dist, min_ind = min((val, idx) for (idx, val) in enumerate(dists))
        if max_dist is not None and max_dist < dist:
            return None
        return min_ind

    # Iterate over segments of a QPolygonF, returns
    # end points of a segment, and index of the second end point
    # The index is where we'll insert the new point, should we need to
    # add one
    class SegmentIterator:
        def __init__(self, poly):
            self._poly = poly
            self._idx = 0

        def __iter__(self):
            return self

        def __next__(self):
            self._idx = self._idx+1
            if self._poly.size() < self._idx: raise StopIteration
            if self._poly.size() == self._idx:
                pta = self._poly[self._idx-1]
                ptb = self._poly[0]
            else:
                pta = self._poly[self._idx-1]
                ptb = self._poly[self._idx]
            a = np.array([[pta.x(), pta.y()], [ptb.x(), ptb.y()]])
            return (a, self._idx)

    def find_insertion_idx(self, pt):
        min_dist=None
        min_idx = 0
        pt = np.array([pt.x(), pt.y()])
        for (seg, idx) in EditablePolygonItem.SegmentIterator(self._polygon):
            dist = point_to_seg_dist(pt, seg)
            if min_dist is None:
                min_dist = dist
                min_idx = idx
            elif dist < min_dist:
                min_dist = dist
                min_idx = idx
        return min_idx

    def mousePressEvent(self, event):
        min_ind = self.find_nearest_point(event.pos().x(), event.pos().y())
        if event.modifiers() == Qt.ControlModifier:
            if min_ind is not None:
                self._polygon.remove(min_ind)
        elif event.modifiers() == Qt.ShiftModifier:
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

    def markReviewed(self):
        """
        This handles marking given polygon as human-reviewed
        """
        self._corrected = True;
        self.updateModel()

    hotkeys = {
        'R': markReviewed,
    }

    def create_new_point(self, pt):
        ptind = self.find_insertion_idx(pt)
        self._polygon.insert(ptind, pt)
        self.updateModel()

    def updateModel(self):
        xns = [str(pt.x()) for pt in self._polygon]
        yns = [str(pt.y()) for pt in self._polygon]
        xn_string = ";".join(xns)
        yn_string = ";".join(yns)
        self._model_item.update({
            self.prefix() + 'xn': xn_string,
            self.prefix() + 'yn': yn_string,
            self.prefix() + 'corrected': self._corrected,
        })

    def paint(self, painter, option, widget=None):
        sloth.items.PolygonItem.paint(self, painter, option, widget)
        for pt in self._polygon:
            painter.drawEllipse(QRectF(pt.x()-1, pt.y()-1, 2, 2))

    def boundingRect(self):
        rect = PolygonItem.boundingRect(self)
        xp1, yp1, xp2, yp2 = rect.getCoords()
        rect.setCoords(xp1-2, yp1-2, xp2+2, yp2+2)
        return rect




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

if whaledetect:
    PLUGINS = [
        WhaleDetectorPlugin
    ]
