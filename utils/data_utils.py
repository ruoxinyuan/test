import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

def load_data(sequences_path, responses_path):
    """
    Load sequences and response data from CSV files.
    """
    df = pd.read_csv(sequences_path)
    labels_df = pd.read_csv(responses_path)
    return df, labels_df

def encode_actions(df, column='action'):
    """
    Encode the specified column using label encoding.
    """
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    return df, label_encoder

def build_sequences(df, id_column='ID', action_column='action', time_column='time'):
    """
    Build (action, time) sequences for each unique ID.
    """
    id_to_sequences = {}
    for _, row in df.iterrows():
        id_ = row[id_column]
        action_time = (row[action_column], row[time_column])
        if id_ not in id_to_sequences:
            id_to_sequences[id_] = []
        id_to_sequences[id_].append(action_time)

    # Convert to PyTorch tensors
    sequences = [torch.tensor(sequence, dtype=torch.float32) for sequence in id_to_sequences.values()]

    lengths = [len(sequence) for sequence in sequences]
    return sequences, lengths

def pad_sequences(sequences, max_length):
    """
    Pad sequences to the same length (max_length) with custom padding values.
    """
    padded_sequences = []

    for sequence in sequences:
        last_time = sequence[-1][1]  # Last time value in the sequence
        padding_value = (-1, last_time)  # Padding value

        pad_len = max_length - sequence.size(0)
        
        if pad_len > 0:
            pad_tensor = torch.tensor([padding_value] * pad_len, dtype=sequence.dtype)
            padded_sequence = torch.cat((sequence, pad_tensor), dim=0)
        else:
            padded_sequence = sequence  # No padding needed
    
        padded_sequences.append(padded_sequence)

    return torch.stack(padded_sequences, dim=0)

def split_data(sequences, labels, lengths, test_size=0.2, random_state=52):
    """
    Split the dataset into training and test sets.
    """
    X_train, X_test, y_train, y_test, lengths_train, lengths_test = train_test_split(
        sequences, labels, lengths, 
        test_size=test_size, 
        random_state=random_state
    )
    return (X_train, y_train, lengths_train), (X_test, y_test, lengths_test)

def create_tensor_datasets(train_data, test_data):
    """
    Create TensorDataset objects for training and testing.
    """
    train_dataset = TensorDataset(*train_data)
    test_dataset = TensorDataset(*test_data)
    return train_dataset, test_dataset
