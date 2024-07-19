from PIL import Image
from transformers import pipeline
import logging
import json

image_path = 'example_images/meong.png'

# Suppress warnings from the transformers library
logging.getLogger('transformers').setLevel(logging.ERROR)

# Load an image
image = Image.open(image_path)

# Allocate a pipeline for object detection
# Explicitly specifying the model and setting the device to GPU (device=0)
object_detector = pipeline('object-detection', model='facebook/detr-resnet-50', device=0)
results = object_detector(image)

# Pretty-print the detection results
print(json.dumps(results, indent=4))
