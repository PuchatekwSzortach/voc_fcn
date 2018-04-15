"""
Module with configuration data
"""

DATA_DIRECTORY = "../../data/VOC2012"

CATEGORIES = [
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor']


IDS_TO_CATEGORIES_MAP = {id: category for id, category in enumerate(CATEGORIES)}
