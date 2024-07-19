import argparse
import json
import logging
import os
from PIL import Image
from transformers import pipeline

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

def perform_object_detection(image_path):
    """
    Perform object detection on an image.
    
    Args:
    image_path (str): Path to the image file.
    
    Returns:
    str: Object detection results in JSON format.
    """
    # Load the image
    image = Image.open(image_path)
    
    # Allocate a pipeline for object detection
    object_detector = pipeline('object-detection', model='facebook/detr-resnet-50', device=0)
    results = object_detector(image)
    
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
    parser.add_argument('--sentences', type=str, nargs='+', help='List of sentences for sentiment analysis')
    parser.add_argument('--sentences-file', type=str, help='Path to a file containing sentences for sentiment analysis')

    args = parser.parse_args()

    # Suppress warnings from the transformers library
    logging.getLogger('transformers').setLevel(logging.ERROR)

    # Handle object detection if an image path is provided
    if args.image:
        detection_results = perform_object_detection(args.image)
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