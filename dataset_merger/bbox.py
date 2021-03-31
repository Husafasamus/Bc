#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import numbers
from typing import Optional, Tuple

import numpy as np


class BBox:
    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        assert min(width, height) >= 0, "width and height must be non-negative"
        
        self._center: np.ndarray = np.asarray((x + width // 2, y + height // 2))
        self._size: np.ndarray = np.asarray((width, height))
    
    @property
    def size(self) -> np.ndarray:
        return self._size
    
    @property
    def center(self) -> np.ndarray:
        return self._center
    
    @size.setter
    def size(self, new_size: np.ndarray) -> None:
        assert (new_size.ndim == 1) and (len(new_size) == 2)
        assert new_size.min() >= 0, "width and height must be non-negative"
        assert issubclass(new_size.dtype.type, numbers.Integral)
        
        self._size = new_size
    
    @staticmethod
    def build_from_center_and_size(
            center: np.ndarray, size: np.ndarray) -> 'BBox':
        assert issubclass(center.dtype.type, numbers.Integral)
        assert issubclass(size.dtype.type, numbers.Integral)
        
        x, y = center - size // 2
        return BBox(x, y, *size)
    
    def as_corners(self) -> np.ndarray:
        xy = self.center - self.size // 2
        return np.concatenate((xy, xy + self.size))
    
    def as_xywh(self, zero_based: bool = True) -> np.ndarray:
        center = self.center if zero_based else self.center + 1
        xy = center - self.size // 2
        return np.concatenate((xy, self.size))
    
    def as_tl_br(self) -> Tuple[np.ndarray, np.ndarray]:
        size_half = self.size // 2
        tl = self.center - size_half
        br = self.center + size_half
        return tl, br
    
    def shift(
            self, center_shift: np.ndarray, in_place=True) -> Optional['BBox']:
        assert (center_shift.ndim == 1) and (len(center_shift) == 2)
        assert issubclass(center_shift.dtype.type, numbers.Integral)
        
        new_center = self._center + center_shift
        if in_place:
            self._center = new_center
            return None
        else:
            return BBox.build_from_center_and_size(new_center, self.size)
    
    def rescale(
            self, width_scale: float, height_scale: float,
            in_place=True) -> Optional['BBox']:
        assert min(width_scale, height_scale) >= 0, \
            "width and height scale factors must be non-negative"
        
        new_size = self.size * np.asarray((width_scale, height_scale))
        new_size = new_size.round().astype(np.int)
        
        if in_place:
            self.size = new_size
            return None
        else:
            return BBox.build_from_center_and_size(self.center, new_size)
    
    def __repr__(self) -> str:
        x, y = self.center - self.size // 2
        width, height = self.size
        return f'{self.__class__.__name__}({x},{y},{width},{height})'


    """
    What I added.
    """
    @staticmethod
    def intersection_of_bboxes(bbx1, bbx2) -> Optional['BBox']:

        bb1_top_left, bb1_bot_right = bbx1.as_tl_br()
        bb2_top_left, bb2_bot_right = bbx2.as_tl_br()

        new_top_left = (max(bb2_top_left[0], bb1_top_left[0]), max(bb2_top_left[1], bb1_top_left[1]))
        new_bot_right = (min(bb1_bot_right[0], bb2_bot_right[0]), min(bb1_bot_right[1], bb2_bot_right[1]))

        if not (new_top_left[0] >= new_bot_right[0] or new_top_left[1] >= new_bot_right[1]):
            return BBox(new_top_left[0], new_top_left[1], new_bot_right[0] - new_top_left[0],
                        new_bot_right[1] - new_top_left[1])
        return None


    @staticmethod
    def intersection_of_list_bboxes(bboxes) -> Optional['BBox']:
        if len(bboxes) == 1:
            return bboxes[0]
        elif len(bboxes) == 0:
            return False

        bbox1 = bboxes[0]
        for bbox in bboxes[1:]:
            bbox1 = BBox.intersection_of_bboxes(bbox1, bbox)
            if not bbox1:
                return False
        return bbox1