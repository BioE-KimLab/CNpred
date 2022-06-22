import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_MKL_REUSE_PRIMITIVE_MEMORY'] = '0'

import numpy as np
from collections import namedtuple

from tensorflow.keras import layers
import nfp

import rdkit.Chem
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcNumHBA, CalcNumHBD

import pandas as pd
import json

class CustomPreprocessor(nfp.SmilesPreprocessor):
    def construct_feature_matrices(self, smiles, train=None):
        features = super(CustomPreprocessor, self).construct_feature_matrices(smiles, train)
        features['mol_features'] = global_features(smiles)
        return features
    
    output_signature = {**nfp.SmilesPreprocessor.output_signature,
                     **{'mol_features': tf.TensorSpec(shape=(2,), dtype=tf.float32) }}

def atom_features(atom):
    atom_type = namedtuple('Atom', ['totalHs', 'symbol', 'aromatic', 'ring_size'])
    return str((atom.GetTotalNumHs(),
                atom.GetSymbol(),
                atom.GetIsAromatic(),
                nfp.preprocessing.features.get_ring_size(atom, max_size=6)
               ))

def bond_features(bond, flipped=False):
    bond_type = namedtuple('Bond', ['bond_type', 'ring_size', 'symbol_1', 'symbol_2'])

    if not flipped:
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()

    else:
        atom1 = bond.GetEndAtom()
        atom2 = bond.GetBeginAtom()

    return str((bond.GetBondType(),
                nfp.preprocessing.features.get_ring_size(bond, max_size=6),
                atom1.GetSymbol(),
                atom2.GetSymbol()
               ))

def global_features(smiles):
    mol = rdkit.Chem.MolFromSmiles(smiles)

    return tf.constant([CalcNumHBA(mol),
                     CalcNumHBD(mol)])

def create_tf_dataset(df, preprocessor, sample_weight = 1.0, train=True): 
    for _, row in df.iterrows():
        inputs = preprocessor.construct_feature_matrices(row['Canonical_SMILES'], train=train)
        if not train:
            one_data_sample_w = 1.0
        elif sample_weight < 1.0:
            if row['Device_tier'] == 1:
                one_data_sample_w = 1.0
            else:
                one_data_sample_w = sample_weight
        else:
            one_data_sample_w = 1.0

        yield ({'atom': inputs['atom'],
                'bond': inputs['bond'],
                'connectivity': inputs['connectivity'],
                'mol_features': global_features(row['Canonical_SMILES'])},
               row['CN'], one_data_sample_w)


def message_block(original_atom_state, original_bond_state,
                 original_global_state, connectivity, features_dim, i):
    
    atom_state = original_atom_state
    bond_state = original_bond_state
    global_state = original_global_state
    
    global_state_update = layers.GlobalAveragePooling1D()(atom_state)
    global_state_update = layers.Dense(features_dim, activation='relu')(global_state_update)
    global_state_update = layers.Dense(features_dim)(global_state_update)
    global_state = layers.Add()([original_global_state, global_state_update])
    
    new_bond_state = nfp.EdgeUpdate()([atom_state, bond_state, connectivity, global_state])
    bond_state = layers.Add()([original_bond_state, new_bond_state])
    
    new_atom_state = nfp.NodeUpdate()([atom_state, bond_state, connectivity, global_state])
    atom_state = layers.Add()([original_atom_state, new_atom_state])
    
    return atom_state, bond_state, global_state

def message_block_no_glob(original_atom_state, original_bond_state, connectivity, features_dim, i):
    atom_state = original_atom_state
    bond_state = original_bond_state

    new_bond_state = nfp.EdgeUpdate()([atom_state, bond_state, connectivity])
    bond_state = layers.Add()([original_bond_state, new_bond_state])

    new_atom_state = nfp.NodeUpdate()([atom_state, bond_state, connectivity])
    atom_state = layers.Add()([original_atom_state, new_atom_state])

    return atom_state, bond_state


