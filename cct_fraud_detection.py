"""
Training script for downstream fradu detection tasks on the CCT dataset.

A DCN model is used to indicate the risk level of a transaction.
"""

import argparse
import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


def set_seed(seed, deterministic=False):
    """
    Set random seeds for reproducibility across all random number generators.
    
    Args:
        seed: Integer seed value, or None for random behavior
        deterministic: Whether to enable deterministic behavior (affects performance)
    
    Returns:
        The actual seed used
    """
    if seed is None:
        # Use time-based seed for random behavior (safe range for all RNGs)
        seed = int(time.time() * 1000) % (2**31)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Only enable deterministic behavior when explicitly requested via --seed
    # as it can significantly impact performance
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    return seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DCN model for fraud detection')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility. If not set, uses time-based seed for random behavior.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Training batch size (default: 1024)')
    return parser.parse_args()


# Parse arguments and set seed at the beginning (only when run as main script)
args = None
actual_seed = None


def use_bin(x, column):
    """
    Map continuous values to discrete bins.
    
    Args:
        x: Input value to be binned
        column: Column name for binning configuration
    
    Returns:
        Bin indices for input values
    """
    global bin_dict, embedding_num_dict
    bins = bin_dict[column]
    num_bins = embedding_num_dict['vocab_size'][column]
    bin_indices = np.digitize(x, bins, right=True)
    bin_indices = np.clip(bin_indices, 1, num_bins)
    assert bin_indices.min() >= 1
    assert bin_indices.max() <= num_bins
    return bin_indices


def load_and_preprocess_data():
    """Load and preprocess data if needed, return the dataframe."""
    # Process data if not already processed
    if not os.path.exists('./train_test_data.csv'):
        # Read raw data
        df = pd.read_csv('save/devsample.csv')
        df['Errors?'] = df['Errors?'].fillna('Unknown')

        cub = pd.read_csv('./cub.csv')

        # Load preprocessed dictionaries
        with open('./encoding_dict.pkl', 'rb') as f:
            encoding_dict = pickle.load(f)
        with open('./bin_dict.pkl', 'rb') as f:
            bin_dict = pickle.load(f)
        embedding_num_dict = torch.load('./embedding_num_dict.pt')
        item_embedding = np.load('./item_embeddings.npy')

        # Create card number mapping
        id2card = dict()
        for k, v in encoding_dict['Card Number'].items():
            id2card[v] = int(k)

        # Map card numbers from cub to df
        cub['Card Number'] = cub['card'].map(id2card)

        # Split user embedding string into 48 float arrays
        emb_cols = cub['usr_emb'].str.split(',', expand=True)
        emb_cols = emb_cols.apply(pd.to_numeric)
        emb_cols.columns = [f'usr_emb_{i+1}' for i in range(48)]
        cub = pd.concat([cub, emb_cols], axis=1)

        # Split behavior embedding string into 74 float arrays
        beh_cols = cub['beh_emb'].str.split(',', expand=True)
        beh_cols = beh_cols.apply(pd.to_numeric)
        beh_cols.columns = [f'beh_emb_{i+1}' for i in range(74)]
        cub = pd.concat([cub, beh_cols], axis=1)

        # Clean up cub dataframe
        cnub = cub.drop(columns=['card', 'usr_emb', 'beh_emb'])

        # Merge embeddings based on Card Number
        df = df.merge(
            cnub,                # Right table to merge
            on='Card Number',    # Join key column
            how='left',          # Keep all rows from left table
            suffixes=('', '_cnub')  # Suffix for duplicate columns
        )
        df.drop(columns=['Card Number'], inplace=True)

        # Remove rows with missing values
        df.dropna(inplace=True)

        # Bin numerical columns
        tmp_dict = dict()
        for k, v in encoding_dict['MCC'].items():
            tmp_dict[int(k)] = v
        encoding_dict['MCC'] = tmp_dict
        
        # Need to make bin_dict and embedding_num_dict available for use_bin
        globals()['bin_dict'] = bin_dict
        globals()['embedding_num_dict'] = embedding_num_dict
        
        df['_amount'] = use_bin(df['amount'].values, 'Amount')
        df['_MCC'] = df['MCC'].map(encoding_dict['MCC'])
        df['_Use Chip'] = df['Use Chip'].map(encoding_dict['Use Chip'])
        seq_columns = ['_amount', '_MCC', '_Use Chip']

        def get_beh_idx(row):
            """
            Get behavior sequence index from row data.
            
            Args:
                row: DataFrame row containing sequence columns
            
            Returns:
                Behavior sequence index or None if not found
            """
            ans = [str(row[i]) for i in seq_columns]
            ans = tuple(ans)
            return encoding_dict['beh_seq'].get(ans, None)
        
        df['token'] = df.apply(get_beh_idx, axis=1)
        df.dropna(inplace=True)
        df.drop(['_amount', '_MCC', '_Use Chip'], axis=1, inplace=True)

        # Calculate cosine similarities
        cosine_similarities = []
        for index, row in df.iterrows():
            # Convert behavior embedding to numpy array
            beh_emb = np.array([row[f'beh_emb_{i+1}'] for i in range(74)], dtype=np.float32)[:item_embedding.shape[1]]
            token_id = int(row['token'])
            emb = item_embedding[token_id]
            
            # Calculate cosine similarity
            dot_product = np.dot(emb, beh_emb)
            norm_emb = np.linalg.norm(emb)
            norm_beh_emb = np.linalg.norm(beh_emb)
            cosine_sim = dot_product / (norm_emb * norm_beh_emb)
            cosine_similarities.append(cosine_sim)

        # Add similarity features
        df['token_cosine_similarity'] = cosine_similarities
        df['token_cosine_similarity_bin'] = pd.cut(
            df['token_cosine_similarity'],
            bins=40,
            labels=False,
            include_lowest=True
        )

        # Remove unnecessary columns
        df.drop(['User', 'Card', 'datetime', 'weight', 'opendate', 'Is Fraud?', 'Use Chip', 'Merchant City'], 
                axis=1, inplace=True)
        print(df.info())
        
        # Save processed data
        df.to_csv('./train_test_data.csv', index=False)
        print("Data saved to ./train_test_data.csv")
    else:
        df = pd.read_csv('./train_test_data.csv')
    
    return df

# =========== Model Definition ===========

class FraudDataset(Dataset):
    """
    Custom dataset for fraud detection model.
    Handles both numeric and categorical features.
    """
    def __init__(self, df):
        # Convert labels to tensor
        self.labels = torch.tensor(df['fraud'].values, dtype=torch.float32)
        
        # Process numeric features - original transaction features
        base_numeric_cols = ['hour_last_txn', 'amount', 'ave30', 'median30', 'std30', 
                            'areaincome', 'income', 'debt', 'FICO Score', 'Num Credit Cards',
                            'creditlimit', 'days_since_open', 'pct_over30ave', 'pct_over30med', 'std_30hist']
        
        # Add user embeddings (48 dimensions)
        usr_emb_cols = [f'usr_emb_{i+1}' for i in range(48)]
        
        # Add behavior embeddings (74 dimensions)
        beh_emb_cols = [f'beh_emb_{i+1}' for i in range(74)]
        
        # Combine all numeric features
        numeric_cols = base_numeric_cols + usr_emb_cols + beh_emb_cols
        self.numeric_data = df[numeric_cols].values.astype(np.float32)
        
        # Standardize numeric features
        self.scaler = StandardScaler()
        self.numeric_data = self.scaler.fit_transform(self.numeric_data)
        
        # Process categorical features
        self.categorical_cols = {
            'MCC': df['MCC'].astype('category').cat.codes.values,
            'Errors?': df['Errors?'].astype('category').cat.codes.values,
            'Card Brand': df['Card Brand'].astype('category').cat.codes.values,
            'Card Type': df['Card Type'].astype('category').cat.codes.values,
            'Has Chip': df['Has Chip'].astype('category').cat.codes.values,
            'Cards Issued': df['Cards Issued'].astype('category').cat.codes.values,
        }
        
        # Record unique value counts for embedding layers
        self.cat_dims = {
            'MCC': len(df['MCC'].unique()),
            'Errors?': len(df['Errors?'].unique()),
            'Card Brand': len(df['Card Brand'].unique()),
            'Card Type': len(df['Card Type'].unique()),
            'Has Chip': len(df['Has Chip'].unique()),
            'Cards Issued': len(df['Cards Issued'].unique()),
        }
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Get single sample from dataset.
        
        Returns:
            tuple: (numeric_features, categorical_features, label)
        """
        numeric = torch.tensor(self.numeric_data[idx], dtype=torch.float32)
        categorical = {k: torch.tensor(v[idx], dtype=torch.long) for k, v in self.categorical_cols.items()}
        label = self.labels[idx]
        return numeric, categorical, label

class DCN(nn.Module):
    """
    Deep & Cross Network for fraud detection.
    Combines cross network and deep network for feature interactions.
    """
    def __init__(self, numeric_dim, cat_dims, embedding_dim=4):
        super(DCN, self).__init__()
        token_cosine_similarity_bin_dim = 32
        
        # Embedding layers for categorical features
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(dim, token_cosine_similarity_bin_dim if col == 'token_cosine_similarity_bin' else embedding_dim)
            for col, dim in cat_dims.items()
        })
        
        # Initialize embedding weights
        for emb in self.embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        
        # Calculate total input dimension
        total_embedding_dim = sum([embedding_dim for _ in cat_dims])
        if 'token_cosine_similarity_bin' in cat_dims.keys():
            total_embedding_dim = total_embedding_dim - embedding_dim + token_cosine_similarity_bin_dim
        total_input_dim = numeric_dim + total_embedding_dim
        print(f"Total input dimension: {total_input_dim}")
        
        # Cross Network for explicit feature interactions
        self.cross_network = CrossNetwork(total_input_dim, num_layers=2)
        
        # Deep Network for implicit feature interactions
        self.deep_network = nn.Sequential(
            nn.Linear(total_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )

        # Combination layer
        self.combine = nn.Sequential(
            nn.Linear(total_input_dim + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        
        # Initialize linear layer weights
        for layer in self.deep_network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
        for layer in self.combine:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
        
    def forward(self, numeric, categorical):
        """
        Forward pass through network.
        
        Args:
            numeric: Numeric features tensor
            categorical: Dictionary of categorical feature tensors
        
        Returns:
            Predicted probability of fraud
        """
        # Process categorical features
        embedded = []
        for col, tensor in categorical.items():
            embedded.append(self.embeddings[col](tensor))
        embedded = torch.cat(embedded, dim=1)
        
        # Combine features
        x = torch.cat([numeric, embedded], dim=1)
        
        # Cross Network
        cross = self.cross_network(x)
        
        # Deep Network
        deep = self.deep_network(x)
        
        # Combine outputs
        combined = torch.cat([cross, deep], dim=1)
        output = torch.sigmoid(self.combine(combined))
        
        return output.squeeze()

class CrossNetwork(nn.Module):
    """
    Cross Network for explicit feature interactions.
    """
    def __init__(self, input_dim, num_layers):
        super(CrossNetwork, self).__init__()
        self.num_layers = num_layers
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, 1))
            for _ in range(num_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim))
            for _ in range(num_layers)
        ])
        
        # Initialize weights
        for weight in self.weights:
            nn.init.xavier_normal_(weight)
        for bias in self.biases:
            nn.init.zeros_(bias)
        
    def forward(self, x):
        """
        Forward pass through cross network.
        
        Args:
            x: Input features tensor
        
        Returns:
            Cross network output
        """
        x0 = x.clone()
        for i in range(self.num_layers):
            cross_term = (x @ self.weights[i]).expand(-1, x.size(1))
            x = x0 * cross_term + self.biases[i] + x
        return x

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    """
    Train fraud detection model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        Trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    best_metrics = {'val_loss': float('inf')}
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # Training loop
        for numeric, categorical, labels in train_loader:
            numeric = numeric.to(device)
            categorical = {k: v.to(device) for k, v in categorical.items()}
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(numeric, categorical)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * numeric.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        tp = 0  # True positives
        fn = 0  # False negatives
        fp = 0  # False positives
        tn = 0  # True negatives
        total = 0  # Total samples
        
        with torch.no_grad():
            for numeric, categorical, labels in val_loader:
                numeric = numeric.to(device)
                categorical = {k: v.to(device) for k, v in categorical.items()}
                labels = labels.to(device)
                
                outputs = model(numeric, categorical)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * numeric.size(0)
                
                predicted = (outputs > 0.5).float()
                # Calculate metrics
                tp += ((predicted == 1) & (labels == 1)).sum().item()
                fn += ((predicted == 0) & (labels == 1)).sum().item()
                fp += ((predicted == 1) & (labels == 0)).sum().item()
                tn += ((predicted == 0) & (labels == 0)).sum().item()
                total += labels.size(0)
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        # Update best metrics
        best_metrics['val_loss'] = min(val_loss, best_metrics.get('val_loss', float('inf')))
        best_metrics['accuracy'] = max(accuracy, best_metrics.get('accuracy', 0.0))
        best_metrics['recall'] = max(recall, best_metrics.get('recall', 0.0))
        best_metrics['precision'] = max(precision, best_metrics.get('precision', 0.0))
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Val Accuracy: {accuracy:.4f}, Val Recall: {recall:.4f}, Val Precision: {precision:.4f}')
    
    # Print final metrics
    print('\nBest Metrics:')
    print(f"Val Loss: {best_metrics['val_loss']:.4f}")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    return model


def main():
    """Main function to run the fraud detection training."""
    global args, actual_seed
    
    # Parse arguments and set seed
    args = parse_args()
    # Enable deterministic mode only when a specific seed is requested
    actual_seed = set_seed(args.seed, deterministic=(args.seed is not None))
    print(f"Using random seed: {actual_seed}")
    
    # Load data
    df = load_and_preprocess_data()
    
    # 1. Load and balance dataset
    fraud_df = df[df['fraud'] == 1]
    non_fraud_df = df[df['fraud'] == 0]

    # Undersample negative class (using the configured seed for reproducibility)
    non_fraud_df = non_fraud_df.sample(n=len(fraud_df), random_state=actual_seed)

    # Create balanced dataset
    balanced_df = pd.concat([fraud_df, non_fraud_df])

    # 2. Split into train/validation sets
    train_df, val_df = train_test_split(balanced_df, test_size=0.2, random_state=actual_seed, stratify=balanced_df['fraud'])

    # 3. Create datasets and data loaders
    train_dataset = FraudDataset(train_df)
    val_dataset = FraudDataset(val_df)

    # Set worker seed function for DataLoader reproducibility
    def worker_init_fn(worker_id):
        worker_seed = actual_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=65536, shuffle=False)

    # 4. Initialize model (seed already set at the beginning)
    numeric_dim = len(train_dataset.numeric_data[0])
    cat_dims = train_dataset.cat_dims

    model = DCN(numeric_dim, cat_dims)

    # 5. Train model
    trained_model = train_model(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr)
    
    return trained_model


if __name__ == '__main__':
    main()
