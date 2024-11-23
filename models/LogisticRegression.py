import torch
import torch.nn as nn
from collections import defaultdict

class LogisticRegressionClassifier(nn.Module):
    def __init__(self, feature_dict):
        """
        Args:
            feature_dict (dict): Precomputed global feature dictionary mapping N-grams to indices.
        """
        super(LogisticRegressionClassifier, self).__init__()
        self.feature_dict = feature_dict
        self.fc = nn.Linear(len(feature_dict) + 2, 1)  # Add 2 for total_time and action_count

    def build_feature_dict(data_loader, max_N=2):
        """
        Builds a global feature dictionary for all N-grams across the dataset.
        Args:
            data_loader (DataLoader): DataLoader containing the dataset.
            max_N (int): Maximum N-gram size.
        Returns:
            dict: A dictionary mapping each N-gram to a unique index.
        """
        feature_set = set()
        
        for batch in data_loader:
            x, _, lengths = batch
            batch_size, _, _ = x.size()

            for i in range(batch_size):
                actions = x[i, :lengths[i], 0].tolist()
                for n in range(1, max_N + 1):
                    for j in range(len(actions) - n + 1):
                        ngram = tuple(actions[j:j + n])
                        feature_set.add(ngram)

        feature_dict = {ngram: idx for idx, ngram in enumerate(sorted(feature_set))}
        return feature_dict

    def N_gram_features(self, x, lengths, max_N=2):
        """
        Extracts N-gram features, total time spent, and action count for each sample.
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, max_length, 2).
            lengths (torch.Tensor): Sequence lengths for each sample, shape (batch_size,).
        Returns:
            torch.Tensor: Unified feature matrix for the batch, with shape (batch_size, num_features).
        """
        batch_size, max_length, _ = x.size()
        feature_matrix = torch.zeros((batch_size, len(self.feature_dict) + 2))  # Add 2 for total_time and action_count

        for i in range(batch_size):
            actions = x[i, :lengths[i], 0].tolist()
            times = x[i, :lengths[i], 1].tolist()

            total_time = max(times) if times else 0.0
            action_count = lengths[i].item()

            ngram_counts = defaultdict(int)
            for n in range(1, max_N + 1):
                for j in range(len(actions) - n + 1):
                    ngram = tuple(actions[j:j + n])
                    if ngram in self.feature_dict:
                        ngram_counts[ngram] += 1

            # Fill feature matrix using the global feature dictionary
            for ngram, count in ngram_counts.items():
                feature_matrix[i, self.feature_dict[ngram]] = count

            feature_matrix[i, -2] = total_time
            feature_matrix[i, -1] = action_count

        return feature_matrix

    def forward(self, x, lengths):
        """
        Forward pass with unified feature extraction.
        """
        x = self.N_gram_features(x, lengths)
        
        logits = self.fc(x)
        return logits
