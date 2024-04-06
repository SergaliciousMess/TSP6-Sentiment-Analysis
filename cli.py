import argparse
from sklearn import model_selection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from SentimentAnalysis import SentimentAnalysis, load_dataset, ground_truth_label_for_text

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Sentiment Analysis CLI')
    parser.add_argument('--architecture', choices=['standard', '1d', 'deep', 'wide', 'parallel', 'dilated', 'attention'], default='standard', help='Select the CNN architecture (default: standard)')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate for training (default: 0.1)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs for training (default: 2000)')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--positive-threshold', type=float, default=0.7, help='Threshold for classifying as Positive (default: 0.7)')
    parser.add_argument('--negative-threshold', type=float, default=0.3, help='Threshold for classifying as Negative (default: 0.3)')
    
    # Add the '--delimiter' argument. Only needed for .txt dataset files
    parser.add_argument('--delimiter', type=str, default=',', help='Delimiter used in the TXT dataset file (default: comma)')

    args = parser.parse_args()

    # Execute functionality based on command-line arguments
    print(f"Architecture: {args.architecture}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Dataset: {args.dataset}")
    print(f"Positive Threshold: {args.positive_threshold}")
    print(f"Negative Threshold: {args.negative_threshold}")

    # Load dataset from the specified file path
    dataset = load_dataset(args.dataset, 'csv', '')  # Hardcoding and assuming CSV format for now

    # Split the dataset into training and testing sets
    train_texts, test_texts = model_selection.train_test_split(dataset, test_size=0.2, random_state=42)

    # Create dataloader for training data
    train_dataloader = DataLoader(train_texts, batch_size=args.batch_size)

    # Initialize SentimentAnalysis with the selected model
    sentiment_analysis = SentimentAnalysis(dataloader=train_dataloader, model_type=args.architecture, learning_rate=args.learning_rate)

    # Train the model
    sentiment_analysis.train_from_dataloader(train_dataloader, epochs=args.epochs)

    # Evaluate the model
    predictions = []
    targets = []
    for text in test_texts:
        prediction = sentiment_analysis.analyze([text[1]])
        predictions.extend(prediction)
        # Assuming that labels are inferred from text or dataset structure
        targets.append(ground_truth_label_for_text(text))

    # # Calculate performance metrics
    # accuracy = accuracy_score(targets, predictions)
    # precision = precision_score(targets, predictions, average='weighted')
    # recall = recall_score(targets, predictions, average='weighted')
    # f1 = f1_score(targets, predictions, average='weighted')

    # # Print performance metrics
    # print(f"Accuracy: {accuracy}")
    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    # print(f"F1 Score: {f1}")

if __name__ == "__main__":
    main()