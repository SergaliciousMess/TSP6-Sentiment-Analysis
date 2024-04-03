import argparse
from . import SentimentAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
class cli:
    def main():
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description='Sentiment Analysis CLI')
        parser.add_argument('--architecture', choices=['standard', '1d', 'deep', 'wide', 'parallel', 'dilated', 'attention'], default='standard', help='Select the CNN architecture (default: standard)')
        parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for training (default: 0.001)')
        parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training (default: 64)')
        parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training (default: 10)')
        parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset file')
        parser.add_argument('--positive-threshold', type=float, default=0.7, help='Threshold for classifying as Positive (default: 0.7)')
        parser.add_argument('--negative-threshold', type=float, default=0.3, help='Threshold for classifying as Negative (default: 0.3)')

        args = parser.parse_args()
    
        # Execute functionality based on command-line arguments
        print(f"Architecture: {args.architecture}")
        print(f"Learning Rate: {args.learning_rate}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Epochs: {args.epochs}")
        print(f"Dataset: {args.dataset}")
        print(f"Positive Threshold: {args.positive_threshold}")
        print(f"Negative Threshold: {args.negative_threshold}")
    
        s = SentimentAnalysis.SentimentAnalysis()
        m = SentimentAnalysis.CNNModel()
    
        # Load dataset from the specified file path
        dataset = s.load_dataset(args.dataset, 'csv')  # Hardcoding and assuming CSV format for now
    
        # Split the dataset into training and testing sets
        train_texts, test_texts = s.train_test_split(dataset, test_size=0.2, random_state=42)
    
        # Initialize SentimentAnalysis with the selected model
        sentiment_analysis = s(dataset=train_texts, model_type=args.architecture, learning_rate=args.learning_rate)
    
        # Train the model
        sentiment_analysis.train(epochs=args.epochs)

        # Evaluate the model
        predictions = []
        targets = []
        for text in test_texts:
            prediction = sentiment_analysis.analyze(text)
            predictions.extend(prediction)
            # Assuming that labels are inferred from text or dataset structure
            targets.extend(s.ground_truth_label_for_text(text))

        # Calculate performance metrics
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, average='weighted')
        recall = recall_score(targets, predictions, average='weighted')
        f1 = f1_score(targets, predictions, average='weighted')

        # Print performance metrics
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

    if __name__ == "__main__":
        main()