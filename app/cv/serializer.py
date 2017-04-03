# -*- coding: utf-8 -*-

"""
serialize bounding boxes from Haar Cascade into json format files
"""
import json

from app.pipeline import folder_traverse


def serialize_json(filename, detector):
    """produce json files """
    annotations = list()
    for (x, y, width, height) in detector:
        annotation = {"class": "rect",
                      "height": int(height),
                      "width": int(width),
                      "x": int(x),
                      "y": int(y)}
        annotations.append(annotation)
    rv = {
        "annotations": annotations,
        "class": "image",
        "filename": str(filename)
    }
    return json.dumps(rv)


def deserialize_json(root_dir, ext=('json')):
    """concatenate and deserialize bounding boxes in json format"""
    bbox_file_structure = folder_traverse(root_dir, ext=ext)
    annotations = list()
    for folder, filelist in bbox_file_structure.items():
        for filename in filelist:
            with open(folder+filename) as f:
                label = json.load(f)
                annotations.append(label)
    # individual json object from nested lists
    annotations_dict = {json_object['filename']: json_object for nested_list
                        in annotations for json_object in nested_list}
    return annotations_dict
