"""
@author SergaliciousMess
@author emyasenc
@author SunOfLife1

Description: an edited verison of primary Python file for the sentiment analysis class,
             originally commited from SergaliciousMess

             The following are further tasks to take into consideration for CNNModel:

             Neutral Sentiment: Define a clear criterion for neutral sentiment.
                                It's a subjective decision and may require additional
                                analysis of the dataset (when we acquire the labeled data).

             Evaluation Metrics: Consider implementing evaluation metrics such as accuracy,
                                 precision, recall, and F1-score to assess model performance
                                 during training and testing.

             Hyperparameter Tuning: Experiment with different hyperparameters such as
                                    learning rate, batch size, and model architecture to
                                    improve model performance.

             Error Handling: Implement error handling mechanisms to gracefully handle
                             exceptions and errors during data processing and model training.
"""

import csv
import json
from enum import Enum

import spacy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator


def ground_truth_label_for_text(text: tuple):
    # Assuming each text sample is a tuple where the label is the first element
    return text[0]


class Architecture(Enum):
    """All acceptable architectures for CNNModel."""
    STANDARD = "standard"
    ONE_DIMENSION = "1d"
    WIDE = "wide"
    DEEP = "deep"
    DILATED = "dilated"
    PARALLEL = "parallel"

    def __str__(self):
        # To work better with argparse
        return self.value


class CNNModel(nn.Module):
    def __init__(self, architecture: Architecture = Architecture.STANDARD, input_size: int = 64, output_size: int = 1):
        super(CNNModel, self).__init__()
        self.architecture = architecture
        # Define the CNN architecture based on the chosen type
        match architecture:
            case Architecture.STANDARD:
                self.model = nn.Sequential(
                    nn.Linear(input_size, output_size),
                    nn.Sigmoid()
                )
            case Architecture.ONE_DIMENSION:
                self.model = nn.Sequential(
                    nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Flatten(),
                    nn.Linear(64 * ((input_size - 2) // 2), output_size),
                    nn.Sigmoid()
                )
            case Architecture.WIDE:
                self.model = nn.Sequential(
                    nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=5),  # Increase out_channels for wider architecture
                    nn.MaxPool1d(kernel_size=2),
                    nn.Flatten(),
                    nn.Linear(128 * ((input_size - 4) // 2), output_size),  # Adjust input_size calculation for wider architecture
                    nn.Sigmoid()
                )
            case Architecture.DEEP:
                self.model = nn.Sequential(
                    nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Flatten(),
                    nn.Linear(64 * ((input_size - 2) // 2), output_size),
                    nn.Sigmoid()
                )
            case Architecture.DILATED:
                self.model = nn.Sequential(
                    nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, dilation=2),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Flatten(),
                    nn.Linear(64 * ((input_size - 4) // 2), output_size),
                    nn.Sigmoid()
                )
            case Architecture.PARALLEL:
                self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3)
                self.conv2 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=5)
                self.conv3 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=7)
                self.pool = nn.MaxPool1d(kernel_size=2)
                self.fc = nn.Linear(32 * ((input_size - 6) // 2), output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.architecture != Architecture.PARALLEL:
            return self.model(x)

        # Apply convolutional operations followed by ReLU activation
        x1 = nn.functional.relu(self.conv1(x))
        x2 = nn.functional.relu(self.conv2(x))
        x3 = nn.functional.relu(self.conv3(x))

        # Apply max pooling to reduce spatial dimensions
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)

        # Concatenate the pooled feature maps along the channel dimension
        x = torch.cat((x1, x2, x3), dim=1)

        # Flatten the concatenated feature maps
        x = x.view(x.size(0), -1)

        # Apply fully connected layer to the flattened feature maps
        x = self.fc(x)

        # Apply sigmoid activation function to the output
        return torch.sigmoid(x)


class TextClassifier(nn.Module):
    def __init__(self, bag: nn.EmbeddingBag, network: CNNModel):
        super(TextClassifier, self).__init__()
        self.bag = bag
        self.network = network

    def forward(self, input: torch.Tensor, offsets: torch.Tensor | None = None) -> torch.Tensor:
        return self.network(self.bag(input, offsets))


class SentimentAnalysis():
    def __init__(
        self,
        dataloader: DataLoader,
        model_type: Architecture = Architecture.STANDARD,
        embedding_width: int = 64,
        learning_rate: float = 0.1,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        loss_function=nn.CrossEntropyLoss(),
        data_type=torch.float32,
        device: str = "cpu"
    ):
        """Initialize necessary SentimentAnalysis components.

        Args:
            * dataloader
            * model_type:
                Desired CNNModel architecture (default: `Architecture.STANDARD`).
            * embedding_width:
                (default: `64`)
            * learning_rate:
                Desired learning rate for optimization (default: `0.1`).
            * optimizer:
                Lambda function for desired PyTorch optimization
                algorithm (default: `torch.optim.Adam`).
            * loss_function:
                Lambda function for desired PyTorch loss calculation
                algorithm (default: `nn.CrossEntropyLoss()`).
            * data_type:
                Desired number format (default: `torch.float32`).
            * device:
                Desired device for computations (default: `"cpu"`).
        """

        try:
            # Initialize nlp and build vocab
            torch.set_default_dtype(data_type)
            torch.set_default_device(device=device)
            self.nlp = spacy.load("en_core_web_sm")
            self.vocab = self.build_vocab(dataloader)

            # Construct model based off of parameters
            bag = nn.EmbeddingBag(num_embeddings=len(self.vocab), embedding_dim=embedding_width, mode="mean")
            bag.to(device=device)
            network = CNNModel(architecture=model_type, input_size=embedding_width, output_size=1)
            network.to(data_type)
            network.to(device=device)
            self.model = TextClassifier(bag=bag, network=network)

            # Store some important variables
            self.data_type = data_type
            self.device = device
            self.embedding_width = embedding_width

            # Initialize optimizer and loss function
            self.learning_rate = learning_rate
            self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
            self.loss_function = loss_function
        except Exception as e:
            print(f"Error occurred during initialization: {e}")
            raise

    def tokenize(self, text: str) -> list[str]:
        # Tokenization logic using spaCy
        try:
            doc = self.nlp(text)
            tokens = [token.lemma_.lower().strip() for token in doc if not token.is_stop and not token.is_punct]
            return tokens
        except Exception as e:
            print(f"Error occurred during tokenization: {e}")
            raise

    def yield_tokens(self, dataloader: DataLoader):
        for batch in dataloader:
            for text in batch[1]:
                yield self.tokenize(text)

    def build_vocab(self, dataloader: DataLoader):
        """Build vocabulary from the tokenized data"""
        try:
            vocab = build_vocab_from_iterator(self.yield_tokens(dataloader), specials=["<unk>"])
            vocab.set_default_index(vocab["<unk>"])
            return vocab
        except Exception as e:
            print(f"Error occurred during vocabulary building: {e}")
            raise

    def text_to_numerical(self, text: str):
        """Convert tokenized text to numerical representation using vocabulary"""
        try:
            numerical_representation = self.vocab(self.tokenize(text))
            return numerical_representation
        except Exception as e:
            print(f"Error occurred during text to numerical conversion: {e}")
            raise
    
    def text_to_tensor_offsets(self, texts: list[str]):
        offsets = [0]  # starting index
        i = 0
        text_tokens = []
        for text in texts:
            numeric = self.text_to_numerical(text)
            length = len(numeric)
            if length > 0:
                text_tokens.append(torch.tensor(numeric))
                i += length
            else:
                text_tokens.append(torch.tensor([0]))
                i += 1
            offsets.append(i)
        
        offsets.pop()
        text_tokens = torch.cat(text_tokens)
        offsets = torch.tensor(offsets,dtype=torch.int32)
        return (text_tokens, offsets)

    def analyze(self, texts: list[str]):
        try:
            self.model.eval()

            tokens_offsets = self.text_to_tensor_offsets(texts)

            output = torch.flatten(self.model(tokens_offsets[0], tokens_offsets[1]))

            positive_threshold = 0.7
            negative_threshold = 0.3
            sentiment_labels = []
            for probability in output:
                if probability >= positive_threshold:
                    sentiment_labels.append("Positive")
                elif probability <= negative_threshold:
                    sentiment_labels.append("Negative")
                else:
                    sentiment_labels.append("Neutral")
            return sentiment_labels
        except Exception as e:
            print(f"Error occurred during sentiment analysis: {e}")
            raise

    # Takes a dataloader as input
    def train_from_dataloader(self, dataloader: DataLoader, epochs=2000):
        try:
            for batch in dataloader:
                labels = batch[0].to(self.data_type)
                tokens_offsets = self.text_to_tensor_offsets(batch[1])

                self.train_from_tensor(labels, tokens_offsets[0], tokens_offsets[1], epochs)
        except Exception as e:
            print(f"Error during training from dataloader: {e}")

    # Takes tensors as input
    def train_from_tensor(self, labels, text_tokens, offsets, epochs=2000):
        try:
            self.model.train()
            for epoch in range(epochs):
                self.optimizer.zero_grad()
                predictions = self.model(text_tokens, offsets)
                predictions = torch.flatten(predictions)
                loss = self.loss_function(predictions, labels)
                loss.backward()
                self.optimizer.step()
        except Exception as e:
            print(f"Error during training from tensor: {e}")


# FIXME(SergaliciousMess): FIX LABEL/DATA ENTRIES FOR JSON AND TXT
def load_dataset(dataset_path, file_format: str, delimiter: str = ","): # Added delimiter in the function parameter for .txt
    """Loads a dataset from a given path. Accepts CSV, JSON, and TXT formats."""
    dataset = []
    match file_format:
        case "csv":
            with open(dataset_path, "r", encoding="utf-8") as file:
                csv_reader = csv.reader(file)
                # Assuming each row represents a sample
                for row in csv_reader:
                    # Assuming the text is in the second column
                    dataset.append((int(row[0]), row[1].strip()))
        case "json":
            with open(dataset_path, "r", encoding="utf-8") as file:
                json_data = json.load(file)
                # Assuming each item represents a sample
                for item in json_data:
                    # Assuming the text is stored under the key "text"
                    dataset.append((item["label"], item["text"].strip()))
        case "txt":
            with open(dataset_path, "r", encoding="utf-8") as file:
                for line in file:
                    parts = line.strip().split(delimiter)
                    label = int(parts[0])
                    text = delimiter.join(parts[1:]).strip('"')  # Remove double quotes around text if present
                    dataset.append((label, text))
        case _:
            raise ValueError("Unsupported dataset format. Supported formats: 'csv', 'json', 'txt'")

    return dataset
