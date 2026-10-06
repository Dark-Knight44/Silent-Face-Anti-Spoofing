# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import argparse
import os

import cv2
import torch

from src.anti_spoof_predict import AntiSpoofPredict
from src.utility import parse_model_name


def main():
    parser = argparse.ArgumentParser(
        description="Test MiniFASNet Anti-Spoofing model"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to the .pth anti-spoofing model"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the face image"
    )

    args = parser.parse_args()

    # Check model
    if not os.path.isfile(args.model):
        raise FileNotFoundError(
            f"Model not found: {args.model}"
        )

    # Check image
    if not os.path.isfile(args.image):
        raise FileNotFoundError(
            f"Image not found: {args.image}"
        )

    # Load image
    image = cv2.imread(args.image)

    if image is None:
        raise ValueError(
            f"Could not read image: {args.image}"
        )

    # Get model information from model filename
    model_name = os.path.basename(args.model)

    h_input, w_input, model_type, scale = parse_model_name(
        model_name
    )

    print("--------------------------------")
    print(f"Model      : {model_name}")
    print(f"Image      : {args.image}")
    print(f"Input size : {w_input} x {h_input}")
    print(f"Model type : {model_type}")
    print("--------------------------------")

    # Resize image to the input size expected by MiniFASNet
    image = cv2.resize(
        image,
        (w_input, h_input)
    )

    # Use GPU if available
    device_id = 0 if torch.cuda.is_available() else -1

    print(
        "Device     : "
        + ("CUDA" if device_id == 0 else "CPU")
    )

    # Load Anti-Spoofing model
    model = AntiSpoofPredict(device_id)

    # Run prediction
    prediction = model.predict(
        image,
        args.model
    )

    # predict() returns a softmax probability array
    probabilities = prediction[0]

    predicted_class = probabilities.argmax()
    confidence = probabilities[predicted_class]

    # MiniFASNet:
    # 0 = SPOOF
    # 1 = REAL
    if predicted_class == 1:
        result = "REAL"
    else:
        result = "SPOOF"

    print("--------------------------------")
    print(f"Prediction : {result}")
    print(f"Confidence : {confidence * 100:.2f}%")
    print("--------------------------------")


if __name__ == "__main__":
    main()