"""Quick script to check ConvE model parameters."""
from pykeen.models import ConvE
from pykeen.triples import TriplesFactory
import torch

# Create a dummy triples factory (needed for ConvE initialization)
# Using minimal example
num_entities = 100
num_relations = 10

# Create dummy triples
dummy_triples = torch.LongTensor([
    [0, 0, 1],
    [1, 1, 2],
    [2, 2, 3]
])

triples_factory = TriplesFactory.from_labeled_triples(
    triples=dummy_triples,
    create_inverse_triples=False
)

# Create ConvE model with typical settings
model = ConvE(
    triples_factory=triples_factory,
    embedding_dim=200,
    output_channels=32
)

print('=' * 60)
print('ConvE Model Parameter Structure')
print('=' * 60)
print(f'\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}')
print(f'\nNumber of parameter tensors: {len(list(model.named_parameters()))}')
print('\nParameter names and shapes:')
print('-' * 60)

for i, (name, param) in enumerate(model.named_parameters(), 1):
    print(f'{i:2d}. {name:40s} {str(list(param.shape)):30s} ({param.numel():,} params)')

print('=' * 60)
