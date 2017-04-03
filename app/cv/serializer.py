# -*- coding: utf-8 -*-

"""
serialize bounding boxes from Haar Cascade into json format files
"""


def serialize_json(filename, detector):
    """produce json files """

    import json
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
