import os
import gin
import logging
from typing import Optional
import mlflow

import torch

from modeling.sequential.output_postprocessors import L2NormEmbeddingPostprocessor, LayerNormEmbeddingPostprocessor
from modeling.sequential.input_features_preprocessors import LearnablePositionalEmbeddingInputFeaturesPreprocessor
from modeling.similarity_utils import get_similarity_function
from modeling.sequential.autoregressive_losses import InBatchNegativesSampler, LocalNegativesSampler
from modeling.sequential.embedding_modules import EmbeddingModule, FeatureEmbeddingModule, LocalCrossFeatureEmbeddingModule, LocalEmbeddingModule, ConcatenateEmbeddingModule

@gin.configurable
def create_negatives_sampler(
        sampling_strategy: str = 'in-batch',
        item_l2_norm: bool = True,
        l2_norm_eps: float = 1e-6,
        max_item_id: int = 3000,
        item_emb = None,
        all_item_ids = None,
):
    if sampling_strategy == "in-batch":
        negatives_sampler = InBatchNegativesSampler(
            l2_norm=item_l2_norm,
            l2_norm_eps=l2_norm_eps,
            dedup_embeddings=True,
        )
        sampling_debug_str = f"in-batch{f'-l2-eps{l2_norm_eps}' if item_l2_norm else ''}-dedup"
    elif sampling_strategy == "local":
        negatives_sampler = LocalNegativesSampler(
            num_items=max_item_id,
            item_emb=item_emb,
            all_item_ids=all_item_ids,
            l2_norm=item_l2_norm,
            l2_norm_eps=l2_norm_eps,
        )
    else:
        raise ValueError(f"Unrecognized sampling strategy {sampling_strategy}.")
    return negatives_sampler

@gin.configurable
def create_model(
    main_module: str = 'HSTU',
    max_sequence_length: int = 256,
    embedding_module_type: str = 'local',
    max_item_id: int = 3000,
    item_embedding_dim: int = 512,
    interaction_module_type: str = 'DotProduct',
    user_embedding_norm: str = 'l2_norm',
    gr_output_length: int = 1,
    main_module_bf16: bool = False,
    use_user_attrs: bool = False,
):
    print(f"modeling.sequential.encoder_utils.create_model.locals(): {locals()}")
    model_debug_str = main_module
    if embedding_module_type == "local":
        embedding_module: EmbeddingModule = LocalEmbeddingModule(
            num_items=max_item_id,
            item_embedding_dim=item_embedding_dim,
        )
    elif embedding_module_type == "feature":
        num_features = 13
        embedding_module: EmbeddingModule = FeatureEmbeddingModule(
            item_embedding_dim=item_embedding_dim,
            input_dim=num_features
        )
    elif embedding_module_type == "local_cross_feature":
        from trainer.tokenizer import action_def, ind_feature_tokenizer,num_feature_values
        embedding_module: EmbeddingModule = LocalCrossFeatureEmbeddingModule(
            num_items=max_item_id,
            item_embedding_dim=item_embedding_dim,
            # num_features = len(action_def),
            # num_feature_values = num_feature_values,
            use_user_attrs = use_user_attrs,
            projector='mean'
        )
    elif embedding_module_type == "ConcatenateEmbeddingModule":
        embedding_module: EmbeddingModule = ConcatenateEmbeddingModule(
            num_items=max_item_id,
            item_embedding_dim=item_embedding_dim,
            use_user_attrs=use_user_attrs,
            feature_embedding_dim="configs/biz-payer-beh-seq/features.yaml"
        )
    else:
        raise ValueError(f"Unknown embedding_module_type {embedding_module_type}")
    model_debug_str += f"-{embedding_module.debug_str()}"

    interaction_module, _ = get_similarity_function(
        module_type=interaction_module_type,
        query_embedding_dim=item_embedding_dim,
        item_embedding_dim=item_embedding_dim,
    )

    assert user_embedding_norm == "l2_norm" or user_embedding_norm == "layer_norm", \
        f"Not implemented for {user_embedding_norm}"
    output_postproc_module = (
        L2NormEmbeddingPostprocessor(
            embedding_dim=item_embedding_dim,
            eps=1e-6,
        ) if user_embedding_norm == "l2_norm" else LayerNormEmbeddingPostprocessor(
            embedding_dim=item_embedding_dim,
            eps=1e-6,
        )
    )
    input_preproc_module = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
        max_sequence_len=max_sequence_length + gr_output_length + 1,
        embedding_dim=item_embedding_dim,
        dropout_rate=0,
    )  # ! actions/ratings are not involved in the input features preprocessor, only item embeddings. !
    
    from modeling.sequential.encoder_utils import get_sequential_encoder  # avoid circular inport
    model = get_sequential_encoder(
        module_type=main_module,
        max_sequence_length=max_sequence_length,
        max_output_length=gr_output_length + 1,
        embedding_module=embedding_module,
        interaction_module=interaction_module,
        input_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        verbose=False
    )
    if main_module_bf16:
        model = model.to(torch.bfloat16)
    return model
