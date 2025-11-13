#!/usr/bin/env python3
"""
Calculate the number of parameters in ConvE model.
"""

def count_conve_parameters(
    num_entities: int,
    num_relations: int,
    embedding_dim: int = 32,
    embedding_height: int = 8,
    embedding_width: int = 4,
    output_channels: int = 32,
    kernel_height: int = 3,
    kernel_width: int = 3
):
    """
    Count parameters in ConvE model.

    Components:
    1. Entity embeddings: num_entities × embedding_dim
    2. Relation embeddings: num_relations × embedding_dim
    3. Conv2d layer: output_channels × in_channels × kernel_height × kernel_width + output_channels (bias)
    4. BatchNorm2d(1): 2 parameters (gamma, beta) - but PyTorch uses 4 (weight, bias, running_mean, running_var)
    5. BatchNorm2d(output_channels): 2 × output_channels
    6. Fully connected layer: flattened_size × embedding_dim + embedding_dim (bias)
    7. BatchNorm1d(embedding_dim): 2 × embedding_dim
    """

    # Calculate flattened size after convolution
    conv_out_height = embedding_height - kernel_height + 1
    conv_out_width = (2 * embedding_width) - kernel_width + 1  # 2* because of concatenation
    flattened_size = output_channels * conv_out_height * conv_out_width

    print("=" * 80)
    print("ConvE Model Architecture")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Entities: {num_entities:,}")
    print(f"  Relations: {num_relations:,}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Embedding height: {embedding_height}")
    print(f"  Embedding width: {embedding_width}")
    print(f"  Output channels: {output_channels}")
    print(f"  Kernel size: ({kernel_height}, {kernel_width})")
    print()

    # Count parameters for each component
    params = {}

    # 1. Entity embeddings
    params['entity_embeddings'] = num_entities * embedding_dim

    # 2. Relation embeddings
    params['relation_embeddings'] = num_relations * embedding_dim

    # 3. Conv2d layer (1 input channel, output_channels output channels)
    conv_weights = output_channels * 1 * kernel_height * kernel_width
    conv_bias = output_channels
    params['conv2d_weights'] = conv_weights
    params['conv2d_bias'] = conv_bias

    # 4. BatchNorm2d(1) - for input
    # BatchNorm has 2 learnable parameters (weight and bias) per channel
    params['bn0'] = 2 * 1  # 1 channel

    # 5. BatchNorm2d(output_channels) - after conv
    params['bn1'] = 2 * output_channels

    # 6. Fully connected layer
    fc_weights = flattened_size * embedding_dim
    fc_bias = embedding_dim
    params['fc_weights'] = fc_weights
    params['fc_bias'] = fc_bias

    # 7. BatchNorm1d(embedding_dim) - after FC
    params['bn2'] = 2 * embedding_dim

    # Print breakdown
    print("Parameter Breakdown:")
    print("-" * 80)
    print(f"1. Entity Embeddings:")
    print(f"   {num_entities:,} × {embedding_dim} = {params['entity_embeddings']:,}")
    print()

    print(f"2. Relation Embeddings:")
    print(f"   {num_relations:,} × {embedding_dim} = {params['relation_embeddings']:,}")
    print()

    print(f"3. Conv2d Layer:")
    print(f"   Weights: {output_channels} × 1 × {kernel_height} × {kernel_width} = {conv_weights:,}")
    print(f"   Bias: {conv_bias:,}")
    print(f"   Total: {conv_weights + conv_bias:,}")
    print()

    print(f"4. BatchNorm2d (input):")
    print(f"   Learnable params: {params['bn0']:,}")
    print()

    print(f"5. BatchNorm2d (after conv):")
    print(f"   Learnable params: {params['bn1']:,}")
    print()

    print(f"6. Fully Connected Layer:")
    print(f"   Input size after conv: {conv_out_height} × {conv_out_width} × {output_channels} = {flattened_size:,}")
    print(f"   Weights: {flattened_size:,} × {embedding_dim} = {fc_weights:,}")
    print(f"   Bias: {fc_bias:,}")
    print(f"   Total: {fc_weights + fc_bias:,}")
    print()

    print(f"7. BatchNorm1d (after FC):")
    print(f"   Learnable params: {params['bn2']:,}")
    print()

    # Calculate totals
    total_embedding_params = params['entity_embeddings'] + params['relation_embeddings']
    total_model_params = (
        params['conv2d_weights'] + params['conv2d_bias'] +
        params['bn0'] + params['bn1'] +
        params['fc_weights'] + params['fc_bias'] +
        params['bn2']
    )
    total_params = total_embedding_params + total_model_params

    print("=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"Total Embedding Parameters: {total_embedding_params:,}")
    print(f"  - Entity embeddings: {params['entity_embeddings']:,}")
    print(f"  - Relation embeddings: {params['relation_embeddings']:,}")
    print()
    print(f"Total Model Parameters (non-embedding): {total_model_params:,}")
    print(f"  - Conv2d: {conv_weights + conv_bias:,}")
    print(f"  - BatchNorms: {params['bn0'] + params['bn1'] + params['bn2']:,}")
    print(f"  - FC layer: {fc_weights + fc_bias:,}")
    print()
    print(f"TOTAL PARAMETERS: {total_params:,}")
    print("=" * 80)

    # Calculate memory usage (assuming float32)
    memory_mb = (total_params * 4) / (1024 * 1024)
    print(f"\nEstimated memory (float32): {memory_mb:.2f} MB")
    print()

    return total_params


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Count ConvE model parameters')
    parser.add_argument('--num-entities', type=int, required=True)
    parser.add_argument('--num-relations', type=int, required=True)
    parser.add_argument('--embedding-dim', type=int, default=32)
    parser.add_argument('--embedding-height', type=int, default=8)
    parser.add_argument('--embedding-width', type=int, default=4)
    parser.add_argument('--output-channels', type=int, default=32)
    parser.add_argument('--kernel-height', type=int, default=3)
    parser.add_argument('--kernel-width', type=int, default=3)

    args = parser.parse_args()

    count_conve_parameters(
        num_entities=args.num_entities,
        num_relations=args.num_relations,
        embedding_dim=args.embedding_dim,
        embedding_height=args.embedding_height,
        embedding_width=args.embedding_width,
        output_channels=args.output_channels,
        kernel_height=args.kernel_height,
        kernel_width=args.kernel_width
    )
