import argparse
import json
import logging
import os
import random
import time
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline
import torch

def analyze_sentiment(texts):
    """
    Analyze sentiment of a list of sentences.
    
    Args:
    texts (list of str): List of sentences to analyze.
    
    Returns:
    dict: Sentiment analysis results in JSON format.
    """
    classifier = pipeline('sentiment-analysis')
    results = [classifier(text) for text in texts]
    return json.dumps(results, indent=4)

def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

def perform_object_detection(image_path, font_path='./arial.ttf'):
    """
    Perform object detection on an image and optionally save the result.
    
    Args:
    image_path (str): Path to the image file.
    font_path (str): Path to the font file (optional).
    
    Returns:
    str: Object detection results in JSON format.
    """
    # Load the image
    image = Image.open(image_path)
    
    # Check if CUDA is available and set device accordingly | Use CPU if cuda is not available
    device = 0 if torch.cuda.is_available() else -1
    
    # Allocate a pipeline for object detection
    object_detector = pipeline('object-detection', model='facebook/detr-resnet-50', device=device)
    results = object_detector(image)
    
    # Prepare result directory and file paths
    result_dir = './result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.splitext(os.path.basename(image_path))[0]
    result_image_path = os.path.join(result_dir, f"{filename}_{timestamp}.jpg")
    result_json_path = os.path.join(result_dir, f"{filename}_{timestamp}.json")
    
    # Draw bounding boxes and labels
    draw = ImageDraw.Draw(image)
    
    # Optionally, load a font
    try:
        font = ImageFont.truetype(font_path, 20) if font_path else ImageFont.load_default()
    except IOError:
        font = ImageFont.load_default()
    
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
    
    # Save the image with annotations
    image.save(result_image_path)
    
    # Save the detection results in JSON format
    with open(result_json_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    
    # Return the detection results in JSON format
    return json.dumps(results, indent=4)

def read_sentences_from_file(file_path):
    """
    Read sentences from a file, one per line.
    
    Args:
    file_path (str): Path to the file containing sentences.
    
    Returns:
    list of str: List of sentences read from the file.
    """
    with open(file_path, 'r') as file:
        sentences = [line.strip() for line in file if line.strip()]
    return sentences

def main():
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Object Detection and Sentiment Analysis')
    parser.add_argument('--image', type=str, help='Path to the image for object detection')
    parser.add_argument('--font', type=str, default='./arial.ttf', help='Path to the font file for annotations')
    parser.add_argument('--sentences', type=str, nargs='+', help='List of sentences for sentiment analysis')
    parser.add_argument('--sentences-file', type=str, help='Path to a file containing sentences for sentiment analysis')

    args = parser.parse_args()

    # Suppress warnings from the transformers library
    logging.getLogger('transformers').setLevel(logging.ERROR)

    # Handle object detection if an image path is provided
    if args.image:
        detection_results = perform_object_detection(args.image, args.font)
        print("Object Detection Results:")
        print(detection_results)
    
    # Handle sentiment analysis if sentences are provided
    if args.sentences:
        sentiment_results = analyze_sentiment(args.sentences)
        print("Sentiment Analysis Results:")
        print(sentiment_results)
    
    # Handle sentiment analysis if a file is provided
    if args.sentences_file:
        sentences_from_file = read_sentences_from_file(args.sentences_file)
        sentiment_results = analyze_sentiment(sentences_from_file)
        print("Sentiment Analysis Results (from file):")
        print(sentiment_results)

if __name__ == '__main__':
    main()