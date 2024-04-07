"""
@author emyasenc
@author SunOfLife1

Description: _desc_
"""

import argparse
from sklearn import metrics, model_selection
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
    print(f"Negative Threshold: {args.negative_threshold}")

    # Load dataset from the specified file path
    # TODO(SunOfLife1): extract dataset file extension from file name
    dataset = load_dataset(args.dataset, 'csv')

    # Split the dataset into training and testing sets
    train_texts, test_texts = model_selection.train_test_split(dataset, test_size=0.2, random_state=42)

    # Create dataloader for training data
    train_dataloader = DataLoader(train_texts, batch_size=args.batch_size)

    # Initialize the model with the inputted arguments
    model = SentimentAnalysis(train_dataloader, model_type=args.architecture, learning_rate=args.learning_rate)

    # Train the model
    model.train_from_dataloader(train_dataloader, epochs=args.epochs)

    # Evaluate the model
    predictions = []
    targets = []
    for text in test_texts:
        prediction = model.analyze([text[1]])
        predictions.extend(prediction)
        # Assuming that labels are inferred from text or dataset structure
        targets.append(ground_truth_label_for_text(text))

    # # Calculate performance metrics
    # accuracy = metrics.accuracy_score(targets, predictions)
    # precision = metrics.precision_score(targets, predictions, average='weighted')
    # recall = metrics.recall_score(targets, predictions, average='weighted')
    # f1 = metrics.f1_score(targets, predictions, average='weighted')

    # # Print performance metrics
    # print(f"Accuracy: {accuracy}")
    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    # print(f"F1 Score: {f1}")


if __name__ == "__main__":
    main()
