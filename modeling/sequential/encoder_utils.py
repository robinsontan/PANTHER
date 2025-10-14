

import os
from typing import Optional

import gin

from modeling.sequential.embedding_modules import EmbeddingModule
from modeling.sequential.hstu import HSTU
from modeling.sequential.input_features_preprocessors import (
    InputFeaturesPreprocessorModule,
)
from modeling.sequential.output_postprocessors import OutputPostprocessorModule
from modeling.sequential.sasrec import SASRec
from modeling.similarity_module import GeneralizedInteractionModule, InteractionModule
from trainer.utils import log_params


@gin.configurable
def sasrec_encoder(
    max_sequence_length: int,
    max_output_length: int,
    embedding_module: EmbeddingModule,
    interaction_module: InteractionModule,
    input_preproc_module: InputFeaturesPreprocessorModule,
    output_postproc_module: OutputPostprocessorModule,
    activation_checkpoint: bool,
    verbose: bool,
    ffn_hidden_dim: int = 64,
    ffn_activation_fn: str = "relu",
    ffn_dropout_rate: float = 0.2,
    num_blocks: int = 2,
    num_heads: int = 1,
    moe:bool = False,
) -> GeneralizedInteractionModule:
    return SASRec(
        embedding_module=embedding_module,
        max_sequence_len=max_sequence_length,
        max_output_len=max_output_length,
        embedding_dim=embedding_module.input_dim,
        ffn_hidden_dim=ffn_hidden_dim,
        ffn_activation_fn=ffn_activation_fn,
        ffn_dropout_rate=ffn_dropout_rate,
        num_blocks=num_blocks,
        num_heads=num_heads,
        similarity_module=interaction_module,
        input_features_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
	    activation_checkpoint=activation_checkpoint,
        verbose=verbose,
        moe=moe
    )

primitives = (bool, str, int, float, type(None))

@gin.configurable
def hstu_encoder(
    max_sequence_length: int,
    max_output_length: int,
    embedding_module: EmbeddingModule,
    interaction_module: InteractionModule,
    input_preproc_module: InputFeaturesPreprocessorModule,
    output_postproc_module: OutputPostprocessorModule,
    activation_checkpoint: bool,
    verbose: bool,
    num_blocks: int = 10,
    num_heads: int = 16,
    dqk: int = 32,
    dv: int = 32,
    linear_dropout_rate: float = 0.0,
    attn_dropout_rate: float = 0.0,
    normalization: str = "rel_bias",
    linear_config: str = "uvqk",
    linear_activation: str = "silu",
    concat_ua: bool = False,
    enable_relative_attention_bias: bool = False,
    experiment_tracker: Optional[dict] = None,
    block_type = ["hstu"],
    scale_up_factor: int = 2,
    residual: bool = True,
    n_patterns: int = None,
    reverse: bool = False,
    num_conv_blocks: int = 1,
) -> GeneralizedInteractionModule:

    if os.environ.get("LOCAL_RANK", None) == '0' and experiment_tracker is not None:
        log_params(experiment_tracker, locals(), prefix='network_config')
        # mlflow.log_params({k:type(v) if type(v) not in primitives else v for k,v in locals().items() if k != 'mlflow_run_id'}, run_id=mlflow_run_id)#, run_id=mlflow.active_run().info.run_id)
    
    return HSTU(
        embedding_module=embedding_module,
        similarity_module=interaction_module,
        input_features_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        max_sequence_len=max_sequence_length,
        max_output_len=max_output_length,
        embedding_dim=embedding_module.item_embedding_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        attention_dim=dqk,
        linear_dim=dv,
        linear_dropout_rate=linear_dropout_rate,
        attn_dropout_rate=attn_dropout_rate,
        linear_config=linear_config,
        linear_activation=linear_activation,
        normalization=normalization,
        concat_ua=concat_ua,
        enable_relative_attention_bias=enable_relative_attention_bias,
        verbose=verbose,
        block_type=block_type,
        scale_up_factor=scale_up_factor,
        residual = residual,
        n_patterns=n_patterns,
        reverse=reverse,
        num_conv_blocks=num_conv_blocks,
    )


@gin.configurable
def get_sequential_encoder(
    module_type: str,
    max_sequence_length: int,
    max_output_length: int,
    embedding_module: EmbeddingModule,
    interaction_module: InteractionModule,
    input_preproc_module: InputFeaturesPreprocessorModule,
    output_postproc_module: OutputPostprocessorModule,
    verbose: bool,
    activation_checkpoint: bool = False,
    block_type = ["hstu"],
    experiment_tracker: Optional[dict] = None,
) -> GeneralizedInteractionModule:
    # mlflow_run_id = mlflow_run_id
    # print("encoder_utils.py: " + str(mlflow_run_id))
    if module_type == "SASRec":
        model = sasrec_encoder(
            max_sequence_length=max_sequence_length,
            max_output_length=max_output_length,
            embedding_module=embedding_module,
            interaction_module=interaction_module,
            input_preproc_module=input_preproc_module,
            output_postproc_module=output_postproc_module,
	        activation_checkpoint=activation_checkpoint,
            verbose=verbose,
        )
    elif module_type == "HSTU":
        model = hstu_encoder(
            max_sequence_length=max_sequence_length,
            max_output_length=max_output_length,
            embedding_module=embedding_module,
            interaction_module=interaction_module,
            input_preproc_module=input_preproc_module,
            output_postproc_module=output_postproc_module,
	        activation_checkpoint=activation_checkpoint,
            verbose=verbose,
            experiment_tracker=experiment_tracker,
            block_type=block_type,
        )
    else:
        raise ValueError(f"Unsupported module_type {module_type}")
    return model
