# This config file is based on attributes suggested by senior
import numpy as np

from openchem.models.Graph2Label import Graph2Label
from openchem.data.feature_data_layer import FeatureDataset
from openchem.modules.encoders.gcn_encoder import GraphCNNEncoder
from openchem.modules.mlp.openchem_mlp import OpenChemMLP
from openchem.data.utils import get_fp
from openchem.utils.utils import identity
from openchem.data.utils import read_smiles_property_file
from openchem.data.utils import save_smiles_property_file

from openchem.utils.graph import Attribute
from openchem.data.graph_data_layer import GraphDataset
import torch.nn as nn
from torch.optim import RMSprop, SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def atomic_number_to_period(atomic_number: int) -> int:
    """
    Convert an atomic number to its corresponding group number.

    Parameters:
    - atomic_number (int): The atomic number of the element.

    Returns:
    - int: The group number that corresponds to the atomic number.
           Returns None if the atomic number is out of the known range.
    """
    if 1 <= atomic_number <= 2:
        return 1
    elif 3 <= atomic_number <= 12:
        return 2
    elif 13 <= atomic_number <= 18:
        return 3
    elif 19 <= atomic_number <= 36:
        return 4
    elif 37 <= atomic_number <= 54:
        return 5
    elif 55 <= atomic_number <= 56 or 72 <= atomic_number <= 86:
        return 6
    elif 57<= atomic_number <= 71:
        return 8 #La 
    elif 87 <= atomic_number <= 88 or 104<=atomic_number<=118:
        return 7
    elif 89<=atomic_number<=103:
        return 9 # Ac
    else:
        return None  # or raise an error for out of range input
    
def atomic_number_to_group(atomic_number: int)->int:
    if atomic_number < 1 or atomic_number > 118:
        return None

    # Define group numbers for elements by their atomic numbers
    groups = [
        1,  # Hydrogen
        18, # Helium
        1,  # Lithium
        2,  # Beryllium
        13, # Boron
        14, # Carbon
        15, # Nitrogen
        16, # Oxygen
        17, # Fluorine
        18, # Neon
        1,  # Sodium
        2,  # Magnesium
        13, # Aluminum
        14, # Silicon
        15, # Phosphorus
        16, # Sulfur
        17, # Chlorine
        18, # Argon
        1,  # Potassium
        2,  # Calcium
        3,  # Scandium
        4,  # Titanium
        5,  # Vanadium
        6,  # Chromium
        7,  # Manganese
        8,  # Iron
        9,  # Cobalt
        10, # Nickel
        11, # Copper
        12, # Zinc
        13, # Gallium
        14, # Germanium
        15, # Arsenic
        16, # Selenium
        17, # Bromine
        18, # Krypton
        1,  # Rubidium
        2,  # Strontium
        3,  # Yttrium
        4,  # Zirconium
        5,  # Niobium
        6,  # Molybdenum
        7,  # Technetium
        8,  # Ruthenium
        9,  # Rhodium
        10, # Palladium
        11, # Silver
        12, # Cadmium
        13, # Indium
        14, # Tin
        15, # Antimony
        16, # Tellurium
        17, # Iodine
        18, # Xenon
        1,  # Cesium
        2,  # Barium
        19, # Lanthanum
        19, # Cerium
        19, # Praseodymium
        19, # Neodymium
        19, # Promethium
        19, # Samarium
        19, # Europium
        19, # Gadolinium
        19, # Terbium
        19, # Dysprosium
        19, # Holmium
        19, # Erbium
        19, # Thulium
        19, # Ytterbium
        19, # Lutetium
        4,  # Hafnium
        5,  # Tantalum
        6,  # Tungsten
        7,  # Rhenium
        8,  # Osmium
        9,  # Iridium
        10, # Platinum
        11, # Gold
        12, # Mercury
        13, # Thallium
        14, # Lead
        15, # Bismuth
        16, # Polonium
        17, # Astatine
        18, # Radon
        1,  # Francium
        2,  # Radium
        20, # Actinium
        20, # Thorium
        20, # Protactinium
        20, # Uranium
        20, # Neptunium
        20, # Plutonium
        20, # Americium
        20, # Curium
        20, # Berkelium
        20, # Californium
        20, # Einsteinium
        20, # Fermium
        20, # Mendelevium
        20, # Nobelium
        20, # Lawrencium
        4,  # Rutherfordium
        5,  # Dubnium
        6,  # Seaborgium
        7,  # Bohrium
        8,  # Hassium
        9,  # Meitnerium
        10, # Darmstadtium
        11, # Roentgenium
        12, # Copernicium
        13, # Nihonium
        14, # Flerovium
        15, # Moscovium
        16, # Livermorium
        17, # Tennessine
        18, # Oganesson
    ]

    return groups[atomic_number - 1]
def get_electronegativity(atomic_number:int)->int:
    # Define a dictionary with electronegativities for elements by their atomic numbers
    electronegativities = {
        1: 2.20, 2: 0.00, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 10: 0.00,
        11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 18: 0.00, 19: 0.82, 20: 1.00,
        21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55, 26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65,
        31: 1.81, 32: 2.01, 33: 2.18, 34: 2.55, 35: 2.96, 36: 0.00, 37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33,
        41: 1.6, 42: 2.16, 43: 1.9, 44: 2.2, 45: 2.28, 46: 2.20, 47: 1.93, 48: 1.69, 49: 1.78, 50: 1.96,
        51: 2.05, 52: 2.1, 53: 2.66, 54: 0.00, 55: 0.79, 56: 0.89, 57: 1.1, 58: 1.12, 59: 1.13, 60: 1.14,
        61: 1.13, 62: 1.17, 63: 1.2, 64: 1.2, 65: 1.22, 66: 1.23, 67: 1.23, 68: 1.24, 69: 1.25, 70: 1.27,
        71: 1.3, 72: 1.3, 73: 1.5, 74: 2.36, 75: 1.9, 76: 2.2, 77: 2.20, 78: 2.28, 79: 2.54, 80: 2.00,
        81: 1.62, 82: 2.33, 83: 2.02, 84: 2.0, 85: 2.2, 86: 0.00, 87: 0.7, 88: 0.89, 89: 1.1, 90: 1.3,
        91: 1.5, 92: 1.38, 93: 1.36, 94: 1.28, 95: 1.13, 96: 1.28, 97: 1.3, 98: 1.3, 99: 1.3, 100: 1.3,
        101: 1.3, 102: 1.3, 103: 1.3, 104: 1.3, 105: 1.3, 106: 1.3, 107: 1.3, 108: 1.3, 109: 1.3, 110: 1.3,
        111: 1.3, 112: 1.3, 113: 1.3, 114: 1.3, 115: 1.3, 116: 1.3, 117: 1.3, 118: 1.3
    }
    
    return electronegativities.get(atomic_number, None)
def get_covalent_radius(atomic_number:int)->int:
    # Define a dictionary with covalent radii for elements by their atomic numbers
    covalent_radii = {
        1: 31, 2: 28, 3: 128, 4: 96, 5: 84, 6: 76, 7: 71, 8: 66, 9: 57, 10: 58,
        11: 166, 12: 141, 13: 121, 14: 111, 15: 107, 16: 105, 17: 102, 18: 106, 19: 203, 20: 176,
        21: 170, 22: 160, 23: 153, 24: 139, 25: 139, 26: 132, 27: 126, 28: 124, 29: 132, 30: 122,
        31: 122, 32: 120, 33: 119, 34: 120, 35: 120, 36: 116, 37: 220, 38: 195, 39: 190, 40: 175,
        41: 164, 42: 154, 43: 147, 44: 146, 45: 142, 46: 139, 47: 145, 48: 144, 49: 142, 50: 139,
        51: 139, 52: 138, 53: 139, 54: 140, 55: 244, 56: 215, 57: 207, 58: 204, 59: 203, 60: 201,
        61: 199, 62: 198, 63: 198, 64: 196, 65: 194, 66: 192, 67: 192, 68: 189, 69: 190, 70: 187,
        71: 187, 72: 175, 73: 170, 74: 162, 75: 151, 76: 144, 77: 141, 78: 136, 79: 136, 80: 132,
        81: 145, 82: 146, 83: 148, 84: 140, 85: 150, 86: 150, 87: 260, 88: 221, 89: 215, 90: 206,
        91: 200, 92: 196, 93: 190, 94: 187, 95: 180, 96: 169, 97: 163, 98: 156, 99: 156, 100: 148,
        101: 143, 102: 138, 103: 136, 104: 157, 105: 149, 106: 143, 107: 141, 108: 134, 109: 129,
        110: 128, 111: 121, 112: 122, 113: 136, 114: 143, 115: 162, 116: 175, 117: 165, 118: 157
    }
    
    return covalent_radii.get(atomic_number, None)


def get_atomic_attributes(atom):
    attr_dict = {}

    atomic_num = atom.GetAtomicNum()
    atomic_mapping = {5: 0, 7: 1, 6: 2, 8: 3, 9: 4, 15: 5, 16: 6, 17: 7, 35: 8,
                    53: 9}
    if atomic_num in atomic_mapping.keys():
        attr_dict['atom_element'] = atomic_mapping[atomic_num]
    else:
        attr_dict['atom_element'] = 10
    attr_dict['period'] = atomic_number_to_period(atomic_num)
    attr_dict['group'] = atomic_number_to_group(atomic_num)
    attr_dict['EN'] = get_electronegativity(atomic_num)
    attr_dict['Covr'] = get_covalent_radius(atomic_num)
    attr_dict['valence'] = atom.GetTotalValence()
    attr_dict['charge'] = atom.GetFormalCharge()
    attr_dict['hybridization'] = atom.GetHybridization().real
    attr_dict['aromatic'] = int(atom.GetIsAromatic())
    return attr_dict

node_attributes = {}
node_attributes['Group number'] = Attribute('node','group',one_hot=True,values=list(range(7)))
node_attributes['Period number'] = Attribute('node','period',one_hot=True,values=list(range(9)))
node_attributes['EN'] = Attribute('node','EN',one_hot=True,values=list(10))
node_attributes['Cov radius'] = Attribute('node','Covr',one_hot=True,values=list(10))
node_attributes['valence'] = Attribute('node', 'valence', one_hot=True, values=[1, 2, 3, 4, 5, 6])
node_attributes['EA'] = Attribute('node','EA',one_hot=True,values=list(10))
node_attributes['Block'] = Attribute('node','Block',one_hot=True,values=[0,1,2,3])
# neglect atomic volume, first IE, second IE
# node_attributes['charge'] = Attribute('node', 'charge', one_hot=True, values=[-1, 0, 1, 2, 3, 4])
# node_attributes['hybridization'] = Attribute('node', 'hybridization',
#                                             one_hot=True, values=[0, 1, 2, 3, 4, 5, 6, 7])
# node_attributes['aromatic'] = Attribute('node', 'aromatic', one_hot=True,
#                                         values=[0, 1])
# node_attributes['atom_element'] = Attribute('node', 'atom_element',
#                                             one_hot=True,
#                                             values=list(range(11)))

data = read_smiles_property_file('../../dataset/allbpwithsmilescleannoheavymolwtlimit.csv',
                                 cols_to_read=[1, 2],
                                 keep_header=False)

smiles = data[1]
labels = np.array(data[:1])
labels = labels.T

X_train, X_test, y_train, y_test = train_test_split(smiles, labels, test_size=0.2,
                                                    random_state=928)

save_smiles_property_file('../../dataset/tmp/train.smi', X_train, y_train)
save_smiles_property_file('../../dataset/tmp/test.smi', X_test, y_test)

train_dataset = GraphDataset(get_atomic_attributes, node_attributes,
                             '../../dataset/tmp/train.smi',
                             delimiter=',', cols_to_read=[1, 0])
test_dataset = GraphDataset(get_atomic_attributes, node_attributes,
                             '../../dataset/tmp/test.smi',
                             delimiter=',', cols_to_read=[1, 0])
predict_dataset = GraphDataset(get_atomic_attributes, node_attributes,
                             '../../dataset/tmp/test.smi',
                             delimiter=',', cols_to_read=[1],return_smiles=True)

model = Graph2Label

# best performance 0.1642 900 eps

model_params = {
    'task': 'regression',
    'random_seed': 928,
    'batch_size': 512,
    'num_epochs': 101,
    'logdir': 'logs/bpgraphcpu',
    'print_every': 20,
    'save_every': 5,
    'train_data_layer': train_dataset,
    'val_data_layer': test_dataset,
    'predict_data_layer': predict_dataset,
    'eval_metrics': r2_score,
    'criterion': nn.MSELoss(),
    'optimizer': Adam,
    'optimizer_params': {
        'lr': 0.0005,
    },
    'lr_scheduler': StepLR,
    'lr_scheduler_params': {
        'step_size': 15,
        'gamma': 0.8
    },
    'encoder': GraphCNNEncoder,
    'encoder_params': {
        'input_size': train_dataset[0]["node_feature_matrix"].shape[1],
        'encoder_dim': 128,
        'n_layers': 5,
        'hidden_size': [128]*5,
    },
    'mlp': OpenChemMLP,
    'mlp_params': {
        'input_size': 128,
        'n_layers': 2,
        'hidden_size': [128, 1],
        'activation': [F.relu, identity]
    }
}
