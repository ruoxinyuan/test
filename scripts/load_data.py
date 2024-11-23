import torch
import sys
import os

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.data_utils import load_data, encode_actions, build_sequences, pad_sequences, split_data

# Ensure the "data" directory exists
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# Load data
actions_path = 'data\cc_data_seqs.csv'
responses_path = 'data\cc_data_responses.csv'
df, labels_df = load_data(actions_path, responses_path)

# Extract response labels
labels = torch.tensor(labels_df['Response'].values, dtype=torch.long)

# Encode the 'action' column
df, label_encoder = encode_actions(df, column='action')

# Build (action, time) sequences
sequences, lengths = build_sequences(df, id_column='ID', action_column='action', time_column='time')

# Pad sequences to the max length
padded_sequences = pad_sequences(sequences, max_length=max(lengths))

# Convert sequence lengths to tensors
lengths = torch.tensor(lengths)

# Split the dataset into training and testing sets
train_data, test_data = split_data(padded_sequences, labels, lengths)
X_train, y_train, lengths_train = train_data
X_test, y_test, lengths_test = test_data

# Save tensors to the "data" folder
torch.save(X_train, os.path.join(output_dir, 'X_train.pt'))
torch.save(y_train, os.path.join(output_dir, 'y_train.pt'))
torch.save(lengths_train, os.path.join(output_dir, 'lengths_train.pt'))

torch.save(X_test, os.path.join(output_dir, 'X_test.pt'))
torch.save(y_test, os.path.join(output_dir, 'y_test.pt'))
torch.save(lengths_test, os.path.join(output_dir, 'lengths_test.pt'))

print("Tensors saved successfully to the 'data' folder.")
