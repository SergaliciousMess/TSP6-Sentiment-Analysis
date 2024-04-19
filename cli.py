"""
@author emyasenc
@author SunOfLife1

Description: _desc_
"""

import argparse
import torch
from sklearn import model_selection
from torch.utils.data import DataLoader
from SentimentAnalysis import (Architecture, SentimentAnalysis,
                               ground_truth_label_for_text, load_dataset)


def parse_args():
    parser = argparse.ArgumentParser(description='Sentiment Analysis CLI')
    parser.add_argument(
        '--architecture',
        type=Architecture,
        choices=Architecture,
        default=Architecture.STANDARD,
        help='Select the CNN architecture (default: standard)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.1,
        help='Learning rate for training (default: 0.1)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for training (default: 64)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=2000,
        help='Number of epochs for training (default: 2000)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to the dataset file'
    )
    parser.add_argument(
        '--delimiter',
        type=str,
        default=',',
        help='Delimiter used for .txt dataset files (default: ",")'
    )
    parser.add_argument(
        '--positive-threshold',
        type=float,
        default=0.7,
        help='Threshold for classifying as Positive (default: 0.7)'
    )
    parser.add_argument(
        '--negative-threshold',
        type=float,
        default=0.3,
        help='Threshold for classifying as Negative (default: 0.3)'
    )

    return parser.parse_args()


def test_accuracy(dataset, batch_size, architecture, learning_rate, epochs, positive_threshold, negative_threshold):
    # Split the dataset into training and testing sets
    train_texts, test_texts = model_selection.train_test_split(dataset, test_size=0.2, random_state=42)

    # Create dataloader for training data
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_dataloader = DataLoader(
        train_texts,
        generator=torch.Generator(device=device),
        batch_size=batch_size
    )

    # Initialize the model with the inputted arguments
    model = SentimentAnalysis(
        train_dataloader,
        model_type=architecture,
        learning_rate=learning_rate
    )

    # Train the model
    model.train_from_dataloader(train_dataloader, epochs=epochs)

    # Evaluate the model
    predictions = model.predictions(t[1] for t in test_texts).tolist()
    targets = [ground_truth_label_for_text(t) for t in test_texts]

    total = 0
    correct = 0
    for i, target in enumerate(targets):
        total += 1

        is_positive = predictions[i] >= positive_threshold
        is_negative = predictions[i] <= negative_threshold
        if (target == 1 and is_positive) or (target == 0 and is_negative):
            correct += 1

    print(f"Score: {correct}/{total} ({round(correct / total * 100, 2)}%)")


def main():
    # Parse command-line arguments
    args = parse_args()

    # Execute functionality based on command-line arguments
    print(f"Architecture: {args.architecture}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Dataset: {args.dataset}")
    print(f"Positive Threshold: {args.positive_threshold}")
    print(f"Negative Threshold: {args.negative_threshold}\n")

    # Load dataset from the specified file path
    # TODO(SunOfLife1): extract dataset file extension from file name
    dataset = load_dataset(args.dataset, 'csv')

    test_accuracy(
        dataset,
        args.batch_size,
        args.architecture,
        args.learning_rate,
        args.epochs,
        args.positive_threshold,
        args.negative_threshold
    )


if __name__ == "__main__":
    main()
