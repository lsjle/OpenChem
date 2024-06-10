import numpy as np

from openchem.models.Smiles2Label import Smiles2Label
from openchem.data.feature_data_layer import SmilesDataset
from openchem.modules.encoders.gcn_encoder import GraphCNNEncoder
from openchem.modules.mlp.openchem_mlp import OpenChemMLP
from openchem.data.utils import get_fp
from openchem.utils.utils import identity
from openchem.data.utils import read_smiles_property_file
from openchem.data.utils import save_smiles_property_file

from openchem.utils.graph import Attribute
from openchem.data.graph_data_layer import GraphDataset
from openchem.criterion.multitask_loss import MultitaskLoss
from openchem.modules.embeddings.basic_embedding import Embedding
from openchem.modules.encoders.rnn_encoder import RNNEncoder
import torch
import torch.nn as nn
from torch.optim import RMSprop, SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

data = read_smiles_property_file('/home/impartialjust/ai/project/ongoing/mppredict/dataset/allbpwithsmilescleannoheavy.csv',
                                 cols_to_read=[1, 3],
                                 keep_header=False)

smiles = data[1]
labels = np.array(data[:1])
labels = labels.T

X_train, X_test, y_train, y_test = train_test_split(smiles, labels, test_size=0.2,
                                                    random_state=928)

save_smiles_property_file('/home/impartialjust/ai/project/ongoing/mppredict/dataset/tmp/train.smi', X_train, y_train)
save_smiles_property_file('/home/impartialjust/ai/project/ongoing/mppredict/dataset/tmp/test.smi', X_test, y_test)

train_dataset = SmilesDataset('./benchmark_datasets/tox21/train.smi',
                              delimiter=',', cols_to_read=[0,1],
                              tokens=tokens, augment=True)
test_dataset = SmilesDataset(get_atomic_attributes, node_attributes,
                             '/home/impartialjust/ai/project/ongoing/mppredict/dataset/tmp/test.smi',
                             delimiter=',', cols_to_read=[0, 1])
predict_dataset = SmilesDataset(get_atomic_attributes, node_attributes,
                             '/home/impartialjust/ai/project/ongoing/mppredict/dataset/tmp/test.smi',
                             delimiter=',', cols_to_read=[0],return_smiles=True)

model = Smiles2Label

model_params = {
    'use_cuda': False,
    'task': 'regression',
    'random_seed': 5,
    'use_clip_grad': True,
    'max_grad_norm': 10.0,
    'batch_size': 256,
    'num_epochs': 21,
    'logdir': 'logs/bprnn',
    'print_every': 5,
    'save_every': 5,
    'train_data_layer': train_dataset,
    'val_data_layer': test_dataset,
    'predict_data_layer': predict_dataset,
    'eval_metrics': r2_score,
    'criterion': nn.MSELoss(),
    'optimizer': RMSprop,
    'optimizer_params': {
        'lr': 0.001,
        },
    'lr_scheduler': StepLR,
    'lr_scheduler_params': {
        'step_size': 10,
        'gamma': 0.8
    },
    'embedding': Embedding,
    'embedding_params': {
        'num_embeddings': train_dataset.num_tokens,
        'embedding_dim': 128,
        'padding_idx': train_dataset.tokens.index(' ')
    },
    'encoder': RNNEncoder,
    'encoder_params': {
        'input_size': 128,
        'layer': "LSTM",
        'encoder_dim': 128,
        'n_layers': 4,
        'dropout': 0.8,
        'is_bidirectional': False
    },
    'mlp': OpenChemMLP,
    'mlp_params': {
        'input_size': 128,
        'n_layers': 2,
        'hidden_size': [128, 12],
        'activation': [F.relu, torch.sigmoid],
        'dropout': 0.0
    }
}
