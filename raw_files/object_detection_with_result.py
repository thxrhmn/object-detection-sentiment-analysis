
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline
import logging
import random

image_path = "example_images/people.jpeg"
result_path = "example_results/people.jpeg"
font_path = "arial.ttf"

# Suppress warnings from the transformers library
logging.getLogger('transformers').setLevel(logging.ERROR)

# Load an image
image = Image.open(image_path)

# Allocate a pipeline for object detection
object_detector = pipeline('object-detection', model='facebook/detr-resnet-50', device=0)
results = object_detector(image)

# Draw bounding boxes and labels
draw = ImageDraw.Draw(image)

# Optionally, load a font
try:
    font = ImageFont.truetype(font_path, 20)
except IOError:
    font = ImageFont.load_default()

def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

for result in results:
    box = result['box']
    xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
    label = result['label']
    score = result['score']
    
    # Generate a random color
    color = random_color()

    # Draw bounding box
    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
    
    # Draw label and score
    text = f"{label} ({score:.2f})"
    text_bbox = draw.textbbox((xmin, ymin), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Draw background rectangle for text
    text_bg_color = (color[0] // 2, color[1] // 2, color[2] // 2)  # Slightly darker background
    draw.rectangle([xmin, ymin - text_height, xmin + text_width, ymin], fill=text_bg_color)
    draw.text((xmin, ymin - text_height), text, fill='white', font=font)

# Save or show the image
image.save(result_path)  # This will open the image in the default image viewer