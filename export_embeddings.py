"""
Script to export embeddings from a trained PANTHER model for downstream tasks.

This script generates two files required by cct_fraud_detection.py:
1. item_embeddings.npy - Item/behavior token embeddings from the model
2. cub.csv - Card-User-Behavior embeddings for each user

Usage:
    python export_embeddings.py \
        --gin_config_file="configs/credit-card-transactions/tf_patternrec_v5.gin" \
        --checkpoint_path="./ckpt/credit-card-transactions/tf_patternrec_v5/ep0_b1000"
"""

import logging
import os
import sys
import traceback

import gin
import numpy as np
import pandas as pd
import torch
from absl import app, flags
from tqdm import tqdm

from eval_tool import ParquetDataset, get_input

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s: %(levelname)s: %(message)s",
)

flags.DEFINE_string("gin_config_file", None, "Path to the config file.")
flags.DEFINE_string("checkpoint_path", None, "Path to the model checkpoint.")
flags.DEFINE_string("output_dir", "./", "Directory to save output files.")
flags.DEFINE_string("local_data_path", "./", "Path to local data directory.")

FLAGS = flags.FLAGS

# Embedding dimensions expected by cct_fraud_detection.py
# usr_emb: 48 dimensions (user attribute embeddings)
# beh_emb: 74 dimensions (behavior sequence embeddings)
USR_EMB_DIM = 48
BEH_EMB_DIM = 74

@gin.configurable
def export_embeddings(
    checkpoint_path: str,
    output_dir: str = "./",
    local_data_path: str = "./",
    max_sequence_length: int = 256,
    item_embedding_dim: int = 50,
    embedding_module_type: str = "ConcatenateEmbeddingModule",
    main_module: str = "SASRec",
    gr_output_length: int = 10,
    block_type=["hstu"],
) -> None:
    """
    Export embeddings from a trained PANTHER model.
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        output_dir: Directory to save output files
        local_data_path: Path to local data directory containing beh_seq_combined.parquet
        max_sequence_length: Maximum sequence length for the model
        item_embedding_dim: Dimension of item embeddings
        embedding_module_type: Type of embedding module used in the model
        main_module: Type of main module used in the model
        gr_output_length: Generative output length
        block_type: Type of blocks used in the model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Validate required files exist
    embedding_dict_path = os.path.join(local_data_path, 'embedding_num_dict.pt')
    if not os.path.exists(embedding_dict_path):
        raise FileNotFoundError(
            f"Required file not found: {embedding_dict_path}. "
            "Please run cct_preprocess.py first to generate this file."
        )
    
    data_path = os.path.join(local_data_path, 'beh_seq_combined.parquet')
    if not os.path.exists(data_path):
        logging.warning(
            f"Data file not found: {data_path}. "
            "Will only export item_embeddings.npy. "
            "cub.csv will not be generated."
        )
    
    # Load embedding configuration
    embedding_num_dict = torch.load(embedding_dict_path)
    max_item_id = embedding_num_dict['vocab_size']['beh_seq']
    
    logging.info(f"Max item ID: {max_item_id}")
    logging.info(f"Embedding module type: {embedding_module_type}")
    
    # Create model
    from modeling.sequential import create_model
    model = create_model(
        max_item_id=max_item_id,
        item_embedding_dim=item_embedding_dim,
        max_sequence_length=max_sequence_length,
        embedding_module_type=embedding_module_type,
        main_module=main_module,
        gr_output_length=gr_output_length,
    )
    
    # Load checkpoint
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                "Please provide a valid path to a trained model checkpoint."
            )
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model_state = state_dict["model_state_dict"]
        # Remove 'module.' prefix if present (from DDP training)
        model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
        
        # Load state dict and log any mismatched keys
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        if missing_keys:
            logging.warning(f"Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
        logging.info("Checkpoint loaded successfully")
    else:
        logging.warning(
            "No checkpoint_path provided. Using untrained model weights. "
            "The exported embeddings may not be meaningful."
        )
    
    model = model.to(device)
    model.eval()
    
    # Export item embeddings
    # Note: Accessing _embedding_module._item_emb is required to get the raw embedding weights
    logging.info("Exporting item embeddings...")
    item_embeddings = model._embedding_module._item_emb.weight.detach().cpu().numpy()
    item_embeddings_path = os.path.join(output_dir, "item_embeddings.npy")
    np.save(item_embeddings_path, item_embeddings)
    logging.info(f"Item embeddings saved to {item_embeddings_path}")
    logging.info(f"Item embeddings shape: {item_embeddings.shape}")
    
    # Generate user behavior embeddings (cub.csv) if data exists
    if not os.path.exists(data_path):
        logging.info("Skipping cub.csv generation due to missing data file.")
        logging.info("Embedding export completed (item_embeddings.npy only)!")
        return
    
    logging.info("Generating user behavior embeddings...")
    dataset = ParquetDataset(data_path, max_length=max_sequence_length + gr_output_length + 1, eval_flag=True)
    
    cub_records = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Processing users"):
            data = dataset[idx]
            
            # Get card number (encoded)
            card_number = data.get('Card Number', idx)
            
            # Prepare batch data (single sample)
            batch_data_dict = {}
            for key, value in data.items():
                if isinstance(value, list):
                    batch_data_dict[key] = torch.tensor([value]).to(device)
                else:
                    batch_data_dict[key] = torch.tensor([[value]]).to(device)
            
            batch_data_dict['seq_length'] = (batch_data_dict['beh_seq'] != 0).sum(dim=1)
            
            try:
                _, seq_features = get_input(embedding_num_dict, batch_data_dict)
                
                # Get input embeddings (user attribute + feature embeddings)
                input_embeddings = model.get_item_embeddings(
                    seq_features.past_ids,
                    features=seq_features.past_payloads.get("features"),
                    user_attrs=seq_features.past_payloads.get("user_attrs"),
                )
                
                # Get sequence embeddings (behavior embeddings)
                seq_embeddings = model(
                    past_lengths=seq_features.past_lengths,
                    past_ids=seq_features.past_ids,
                    past_embeddings=input_embeddings,
                    past_payloads=seq_features.past_payloads,
                    return_cache_states=False,
                )
                
                # Handle different model output formats
                if isinstance(seq_embeddings, tuple):
                    seq_embeddings = seq_embeddings[0]
                
                # Average pooling over sequence dimension for user embedding
                mask = seq_features.past_ids != 0
                seq_len = mask.sum(dim=1, keepdim=True).clamp(min=1)
                usr_emb = (seq_embeddings * mask.unsqueeze(-1)).sum(dim=1) / seq_len
                usr_emb = usr_emb.squeeze(0).cpu().numpy()
                
                # Get the last behavior embedding
                last_idx = seq_features.past_lengths[0].item() - 1
                if last_idx >= 0:
                    beh_emb = seq_embeddings[0, last_idx, :].cpu().numpy()
                else:
                    beh_emb = seq_embeddings[0, 0, :].cpu().numpy()
                
                # Format embeddings as comma-separated strings
                # Dimensions defined by cct_fraud_detection.py expectations
                usr_emb_str = ",".join([f"{x:.6f}" for x in usr_emb[:USR_EMB_DIM]])
                beh_emb_str = ",".join([f"{x:.6f}" for x in beh_emb[:BEH_EMB_DIM]])
                
                cub_records.append({
                    "card": card_number,
                    "usr_emb": usr_emb_str,
                    "beh_emb": beh_emb_str,
                })
            except (RuntimeError, ValueError) as e:
                logging.warning(f"Error processing user {idx}: {e}")
                logging.debug(traceback.format_exc())
                continue
    
    # Save cub.csv
    if cub_records:
        cub_df = pd.DataFrame(cub_records)
        cub_path = os.path.join(output_dir, "cub.csv")
        cub_df.to_csv(cub_path, index=False)
        logging.info(f"User behavior embeddings saved to {cub_path}")
        logging.info(f"Total users processed: {len(cub_records)}")
    else:
        logging.warning("No user behavior embeddings were generated.")
    
    logging.info("Embedding export completed!")


def main(argv):
    if FLAGS.gin_config_file is not None:
        gin_path = os.path.abspath(FLAGS.gin_config_file)
        
        # Extract config directory from gin config file path
        # Expected format: configs/{config_group}/{config}.gin
        gin_dir = os.path.dirname(gin_path)
        if os.path.isdir(gin_dir):
            config_dir = gin_dir
        else:
            # Fallback: try to construct from current directory
            config_group = os.path.basename(os.path.dirname(FLAGS.gin_config_file))
            config_dir = os.path.join(os.getcwd(), 'configs', config_group)
        
        if os.path.isdir(config_dir):
            old_directory = os.getcwd()
            os.chdir(config_dir)
            gin.parse_config_file(gin_path)
            os.chdir(old_directory)
        else:
            # Parse gin config from current directory if config_dir doesn't exist
            logging.warning(f"Config directory not found: {config_dir}. Parsing gin config from current directory.")
            gin.parse_config_file(gin_path)
    
    export_embeddings(
        checkpoint_path=FLAGS.checkpoint_path,
        output_dir=FLAGS.output_dir,
        local_data_path=FLAGS.local_data_path,
    )


if __name__ == "__main__":
    app.run(main)
