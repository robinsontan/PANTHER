"""
Tokenizer module for PANTHER behavioral sequence modeling.

Implements Structured Tokenization mechanism from the PANTHER paper:
- Behavioral tokens as Cartesian products of multi-dimensional attributes
- Vocabulary compression (from ~2M to 60k tokens with >96% coverage)
- Continuous feature bucketization using quantiles
"""

import os
import pickle
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


class BehavioralTokenizer:
    """
    Tokenizer for behavioral sequences with Cartesian product tokenization.
    
    Special tokens:
    - PAD_ID = 0: Padding token
    - UNK_ID = 1: Unknown token for out-of-vocabulary combinations
    """
    
    PAD_ID = 0
    UNK_ID = 1
    
    def __init__(self, vocab_dict: Dict[str, int], reverse_vocab: Optional[Dict[int, str]] = None):
        """
        Initialize tokenizer with vocabulary.
        
        Args:
            vocab_dict: Mapping from token strings to token IDs
            reverse_vocab: Optional reverse mapping from token IDs to strings
        """
        self._vocab = vocab_dict
        if reverse_vocab is None:
            self._reverse_vocab = {v: k for k, v in vocab_dict.items()}
        else:
            self._reverse_vocab = reverse_vocab
    
    def get(self, key: str, default=None):
        """Get token ID for a token string."""
        return self._vocab.get(key, default)
    
    def __contains__(self, key: str) -> bool:
        """Check if token string exists in vocabulary."""
        return key in self._vocab
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self._vocab)
    
    def items(self):
        """Return items from vocabulary."""
        return self._vocab.items()
    
    def keys(self):
        """Return keys from vocabulary."""
        return self._vocab.keys()
    
    def values(self):
        """Return values from vocabulary."""
        return self._vocab.values()
    
    def reverse_lookup(self, token_id: int) -> str:
        """Get token string for a token ID."""
        return self._reverse_vocab.get(token_id, f"<UNK:{token_id}>")


# Global tokenizer configuration - these will be set by get_tokenizer()
feature_columns: List[str] = []
attr_columns: List[str] = []
continuous_attrs: List[str] = []
ignore_loss_attrs: List[str] = []

ind_ind_feature_tokenizer: Dict[str, int] = {}
ind_feature_tokenizer: Dict[str, Dict] = {}
user_attr_tokenizer: Dict[str, int] = {}
num_feature_values: Dict[str, int] = {}
action_def: OrderedDict = OrderedDict()


def bucketize_amount(amounts: List[Union[int, float]], 
                     num_buckets: int = 100,
                     bins: Optional[List[float]] = None) -> List[str]:
    """
    Bucketize continuous amount values using quantile or pre-computed bins.
    
    Args:
        amounts: List of amount values to bucketize
        num_buckets: Number of buckets (used if bins not provided)
        bins: Pre-computed bin boundaries
        
    Returns:
        List of bucket indices as strings
    """
    if not amounts:
        return []
    
    # Convert all amounts to float, handling various types
    amounts_array = np.array([
        float(a) if (a is not None and (isinstance(a, (int, float, str)) and str(a) != '')) else 0.0 
        for a in amounts
    ])
    
    if bins is not None:
        # Use pre-computed bins
        bin_indices = np.digitize(amounts_array, bins, right=True)
        # Clip to valid range [1, len(bins)]
        bin_indices = np.clip(bin_indices, 1, len(bins))
    else:
        # Compute quantile-based bins on-the-fly
        if len(amounts_array) < num_buckets:
            # Not enough data for all buckets
            unique_values = np.unique(amounts_array)
            bins = np.sort(unique_values)
            bin_indices = np.digitize(amounts_array, bins, right=True) + 1
            bin_indices = np.clip(bin_indices, 1, len(bins) + 1)
        else:
            # Use quantile binning
            quantiles = np.linspace(0, 100, num_buckets + 1)
            bins = np.percentile(amounts_array, quantiles)
            bins = np.unique(bins)  # Remove duplicates
            bin_indices = np.digitize(amounts_array, bins, right=True) + 1
            bin_indices = np.clip(bin_indices, 1, len(bins))
    
    return [str(int(idx)) for idx in bin_indices]


def tokenize(beh_tokens: Dict[str, List[str]], 
             return_token_str: bool = False,
             tokenizer: Optional[BehavioralTokenizer] = None) -> Tuple[List[int], Optional[List[str]]]:
    """
    Tokenize behavioral sequences using Cartesian product combination.
    
    Converts multi-dimensional feature sequences into single token IDs by:
    1. Combining features at each position using Cartesian product
    2. Looking up the combined token in vocabulary
    3. Returning UNK_ID if combination not found
    
    Args:
        beh_tokens: OrderedDict mapping feature names to sequences of values
        return_token_str: If True, also return token string representations
        tokenizer: BehavioralTokenizer instance with vocabulary
        
    Returns:
        Tuple of (token_ids, token_strings) where token_strings is None if not requested
    """
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided")
    
    # Get sequence length from first feature
    seq_len = len(next(iter(beh_tokens.values())))
    
    # Verify all features have same length
    for feature_name, feature_values in beh_tokens.items():
        if len(feature_values) != seq_len:
            raise ValueError(f"Feature {feature_name} has length {len(feature_values)}, expected {seq_len}")
    
    token_ids = []
    token_strings = [] if return_token_str else None
    
    # Combine features at each position
    for i in range(seq_len):
        # Create token string by combining all feature values at position i
        # Format: (value1, value2, value3, ...)
        values = tuple(beh_tokens[feat][i] for feat in beh_tokens.keys())
        token_str = str(values)
        
        # Look up in vocabulary
        token_id = tokenizer.get(token_str, BehavioralTokenizer.UNK_ID)
        token_ids.append(token_id)
        
        if return_token_str:
            token_strings.append(token_str)
    
    return token_ids, token_strings


def de_tokenize(token_ids: List[int], 
                tokenizer: Optional[BehavioralTokenizer] = None,
                de_tokenizer: Optional[BehavioralTokenizer] = None) -> List[str]:
    """
    Convert token IDs back to token string representations.
    
    Args:
        token_ids: List of token IDs to decode
        tokenizer: BehavioralTokenizer instance (can be None if de_tokenizer provided)
        de_tokenizer: Alias for tokenizer parameter (for backward compatibility)
        
    Returns:
        List of token string representations
    """
    tok = de_tokenizer if de_tokenizer is not None else tokenizer
    if tok is None:
        raise ValueError("Tokenizer must be provided")
    
    return [tok.reverse_lookup(token_id) for token_id in token_ids]


def get_tokenizer(path: str = "prune99") -> BehavioralTokenizer:
    """
    Load pre-computed tokenizer assets and return configured BehavioralTokenizer.
    
    The tokenizer loads three key files:
    1. encoding_dict.pkl: Main vocabulary mapping
    2. bin_dict.pkl: Binning boundaries for continuous features
    3. embedding_num_dict.pt: Embedding configuration
    
    Args:
        path: Path prefix for tokenizer assets (e.g., "prune99", "prune95")
        
    Returns:
        BehavioralTokenizer instance with loaded vocabulary
    """
    global feature_columns, attr_columns, continuous_attrs, ignore_loss_attrs
    global ind_ind_feature_tokenizer, ind_feature_tokenizer, user_attr_tokenizer
    global num_feature_values, action_def
    
    # Try to load encoding_dict.pkl
    encoding_dict_path = f"{path}/encoding_dict.pkl" if not path.endswith('.pkl') else path
    if not os.path.exists(encoding_dict_path):
        # Try alternative paths
        for alt_path in ['./encoding_dict.pkl', 'encoding_dict.pkl', f'./{path}_encoding_dict.pkl']:
            if os.path.exists(alt_path):
                encoding_dict_path = alt_path
                break
    
    # Load encoding dictionary
    if os.path.exists(encoding_dict_path):
        with open(encoding_dict_path, 'rb') as f:
            encoding_dict = pickle.load(f)
    else:
        # Create minimal default vocabulary if no file found
        encoding_dict = {
            'beh_seq': {str(i): i+1 for i in range(3000)},  # Default vocabulary
        }
    
    # Try to load bin_dict.pkl for continuous feature binning
    if path.endswith('.pkl'):
        # If path is a specific file, look for bin_dict in same directory
        bin_dict_path = path.replace('encoding_dict.pkl', 'bin_dict.pkl')
    else:
        bin_dict_path = f"{path}/bin_dict.pkl"
    
    if not os.path.exists(bin_dict_path):
        for alt_path in ['./bin_dict.pkl', 'bin_dict.pkl', f'./{path}_bin_dict.pkl']:
            if os.path.exists(alt_path):
                bin_dict_path = alt_path
                break
    
    bin_dict = {}
    if os.path.exists(bin_dict_path):
        with open(bin_dict_path, 'rb') as f:
            bin_dict = pickle.load(f)
    
    # Try to load embedding_num_dict.pt for configuration
    embedding_dict_path = f"{path}/embedding_num_dict.pt" if not path.endswith('.pt') else path
    if not os.path.exists(embedding_dict_path):
        for alt_path in ['./embedding_num_dict.pt', 'embedding_num_dict.pt', f'./{path}_embedding_num_dict.pt']:
            if os.path.exists(alt_path):
                embedding_dict_path = alt_path
                break
    
    if os.path.exists(embedding_dict_path):
        embedding_config = torch.load(embedding_dict_path)
        seq_columns = embedding_config.get('seq_columns', ['Amount', 'MCC', 'Use Chip'])
        element_columns = embedding_config.get('element_column', [])
    else:
        # Default configuration based on cct_preprocess.py
        seq_columns = ['Amount', 'MCC', 'Use Chip']
        element_columns = ['Card Number', 'Card Brand', 'Card Type', 'Has Chip', 
                          'Cards Issued', 'Num Credit Cards', 'Current Age',
                          'Gender', 'Credit Limit', 'FICO Score', 
                          'Per Capita Income - Zipcode', 'Yearly Income - Person']
    
    # Map to UHB column names (User Habit Behavior)
    # The UHB prefix is used in the actual data
    feature_mapping = {
        'Amount': 'uhb_amount_bucket_seq',
        'MCC': 'uhb_mcc_seq',
        'Use Chip': 'uhb_usechip_seq',
    }
    
    # Configure feature columns (for Cartesian product tokenization)
    # IMPORTANT: Modify lists in place, don't reassign, so imported references stay valid
    feature_columns.clear()
    feature_columns.extend([feature_mapping.get(col, col.lower().replace(' ', '_')) 
                           for col in seq_columns])
    
    # Configure attribute columns (user-level static attributes)
    attr_columns.clear()
    attr_columns.extend([col.lower().replace(' ', '_').replace('-', '_') 
                        for col in element_columns])
    
    # Continuous attributes that need bucketization
    continuous_attrs.clear()
    continuous_attrs.extend(['uhb_amount_bucket_seq', 'credit_limit', 
                            'per_capita_income_zipcode', 'yearly_income_person'])
    
    # Features to ignore during loss computation
    ignore_loss_attrs.clear()
    ignore_loss_attrs.extend(['uhb_timestamp_seq', 'uhb_mch_sec_risklvl_seq'])
    
    # Build individual feature tokenizers
    ind_feature_tokenizer.clear()
    for feature_name, feature_vocab in encoding_dict.items():
        if feature_name != 'beh_seq' and isinstance(feature_vocab, dict):
            ind_feature_tokenizer[feature_name] = feature_vocab
    
    # Build flat tokenizer for individual features (format: "feature_value")
    ind_ind_feature_tokenizer.clear()
    for feature_name, feature_vocab in ind_feature_tokenizer.items():
        for value, idx in feature_vocab.items():
            key = f"{feature_name}_{value}"
            ind_ind_feature_tokenizer[key] = idx
    
    # Build user attribute tokenizer
    user_attr_tokenizer.clear()
    for attr_name in attr_columns:
        if attr_name in encoding_dict:
            for value, idx in encoding_dict[attr_name].items():
                key = f"{attr_name}_{value}"
                user_attr_tokenizer[key] = idx
    
    # Calculate number of possible values per feature
    num_feature_values.clear()
    num_feature_values.update({
        feature_name: len(feature_vocab) 
        for feature_name, feature_vocab in ind_feature_tokenizer.items()
    })
    
    # Define action structure (behavioral token structure)
    action_def.clear()
    for col in feature_columns:
        action_def[col] = num_feature_values.get(col, 100)
    
    # Get main behavioral sequence vocabulary
    if 'beh_seq' in encoding_dict:
        vocab_dict = encoding_dict['beh_seq']
    else:
        # Fallback: create vocabulary from Cartesian product
        vocab_dict = {str(i): i+1 for i in range(3000)}
    
    # Create tokenizer instance
    tokenizer = BehavioralTokenizer(vocab_dict)
    
    return tokenizer
