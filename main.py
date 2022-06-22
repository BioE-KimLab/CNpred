import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    device = "/gpu:0"
else:
    device = "/cpu:0"

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_MKL_REUSE_PRIMITIVE_MEMORY'] = '0'

import numpy as np
import pandas as pd
from tensorflow.keras import layers
from gnn import *
import nfp
import json 
import sys

from argparse import ArgumentParser
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.model_selection import KFold
#import shap

def main(args):
    data = pd.read_csv('data/CN_merged_210831.csv')
    data_tier1 = data[data.Device_tier == 1] 
    data_tier23 = data[data.Device_tier > 1]

    if args.tier1_only:
        data = data_tier1

    if args.split_option == 0: 
        # just a random 8:1:1 split
        train = data.sample(frac=.8, random_state=args.random_seed)
        valid = data[~data.index.isin(train.index)].sample(frac=.5, random_state=args.random_seed)
        test = data[~data.index.isin(train.index) & ~data.index.isin(valid.index)]

        if args.fold_number != -1 and args.tier1_only: # tier1only + K-fold CV
            kfold = KFold(n_splits=10, shuffle=True, random_state = 1)
            split_tier1 = list(kfold.split(data))[args.fold_number]
            train_index, valid_index = split_tier1
            train_kfold_tier1 = data.iloc[train_index]
            valid_kfold_tier1 = data.iloc[valid_index]

            train['Train/Valid/Test'] = 'Train'
            valid['Train/Valid/Test'] = 'Valid'

    elif args.split_option == 1: 
        # 8:1:1 split, Training set: Tier1,2,3, validation/test set: Tier 1 only
        # training set: 317(Tier 1), 187(Tier 2,3), validation/test set: 63/63
        valid = data_tier1.sample( n=int(len(data_tier1) * 0.1) , random_state=args.random_seed )
        test =  data_tier1[~data_tier1.index.isin(valid.index)].sample( n=int(len(data_tier1) * 0.1) , random_state=args.random_seed )
        train = data_tier1[~data_tier1.index.isin(valid.index) & ~data_tier1.index.isin(test.index)]
        train = pd.concat([train, data_tier23], ignore_index=True)
    elif args.split_option == 2:
        # 8:1:1 split from Tier 1 + 8:1:1 split from Tier 2,3
        train_tier1 = data_tier1.sample(frac=.8, random_state=args.random_seed)
        valid_tier1 = data_tier1[~data_tier1.index.isin(train_tier1.index)].sample(frac=.5, random_state=args.random_seed)
        test_tier1 =  data_tier1[~data_tier1.index.isin(train_tier1.index) & ~data_tier1.index.isin(valid_tier1.index)] 

        train_tier23 = data_tier23.sample(frac=.8, random_state=args.random_seed)
        valid_tier23 = data_tier23[~data_tier23.index.isin(train_tier23.index)].sample(frac=.5, random_state=args.random_seed)
        test_tier23 =  data_tier23[~data_tier23.index.isin(train_tier23.index) & ~data_tier23.index.isin(valid_tier23.index)] 
        
        train = pd.concat([train_tier1, train_tier23], ignore_index=True)
        valid = pd.concat([valid_tier1, valid_tier23], ignore_index=True)
        test = pd.concat([test_tier1, test_tier23], ignore_index=True)

        if args.fold_number != -1:
            data_for_kfold_tier1 = pd.concat([ train_tier1,valid_tier1,test_tier1 ], ignore_index=True)
            data_for_kfold_tier23 = pd.concat([ train_tier23,valid_tier23,test_tier23 ], ignore_index=True)

            kfold = KFold(n_splits=10, shuffle=True, random_state = 1)
            split_tier1 = list(kfold.split(data_for_kfold_tier1))[args.fold_number]
            train_index, valid_index = split_tier1
            train_kfold_tier1 = data_for_kfold_tier1.iloc[train_index]
            valid_kfold_tier1 = data_for_kfold_tier1.iloc[valid_index]

            split_tier23 = list(kfold.split(data_for_kfold_tier23))[args.fold_number]
            train_index, valid_index = split_tier23
            train_kfold_tier23 = data_for_kfold_tier23.iloc[train_index]
            valid_kfold_tier23 = data_for_kfold_tier23.iloc[valid_index]

            train = pd.concat([train_kfold_tier1,train_kfold_tier23], ignore_index=True)
            valid = pd.concat([valid_kfold_tier1,valid_kfold_tier23], ignore_index=True)
        
            train['Train/Valid/Test'] = 'Train'
            valid['Train/Valid/Test'] = 'Valid'

    preprocessor = CustomPreprocessor(
        explicit_hs=False,
        atom_features=atom_features,
        bond_features=bond_features)
            
    output_signature = (preprocessor.output_signature,
                        tf.TensorSpec(shape=(), dtype=tf.float32),
                        tf.TensorSpec(shape=(), dtype=tf.float32))
 
    print(f"Atom classes before: {preprocessor.atom_classes} (includes 'none' and 'missing' classes)")
    print(f"Bond classes before: {preprocessor.bond_classes} (includes 'none' and 'missing' classes)")

    for smiles in train.Canonical_SMILES:
        preprocessor.construct_feature_matrices(smiles, train=True)
    print(f'Atom classes after: {preprocessor.atom_classes}')
    print(f'Bond classes after: {preprocessor.bond_classes}')

    train_data = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset(train, preprocessor, args.sample_weight, True), output_signature=output_signature)\
        .cache().shuffle(buffer_size=1000)\
        .padded_batch(batch_size=args.batchsize)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    valid_data = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset(valid, preprocessor, args.sample_weight, False), output_signature=output_signature)\
        .cache()\
        .padded_batch(batch_size=args.batchsize)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    test_data = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset(test, preprocessor, args.sample_weight, False), output_signature=output_signature)\
        .cache()\
        .padded_batch(batch_size=args.batchsize)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    atom_Input = layers.Input(shape=[None], dtype=tf.int32, name='atom')
    bond_Input = layers.Input(shape=[None], dtype=tf.int32, name='bond')
    connectivity_Input = layers.Input(shape=[None, 2], dtype=tf.int32, name='connectivity')
    global_Input = layers.Input(shape=[2], dtype=tf.float32, name='mol_features')

    features_dim = args.num_hidden
    num_messages = args.layers

    atom_state = layers.Embedding(preprocessor.atom_classes, features_dim,
                                  name='atom_embedding', mask_zero=True,
                                  embeddings_regularizer='l2')(atom_Input)
    bond_state = layers.Embedding(preprocessor.bond_classes, features_dim,
                                  name='bond_embedding', mask_zero=True,
                                  embeddings_regularizer='l2')(bond_Input)

    if args.no_glob_feat:
        atom_mean = layers.Embedding(preprocessor.atom_classes, 1,
                                     name='atom_mean', mask_zero=True,
                                     embeddings_regularizer='l2')(atom_Input)

        for i in range(num_messages):
            atom_state, bond_state = message_block_no_glob(atom_state, bond_state,
                                                        connectivity_Input, features_dim, i)

        atom_state = layers.Add()([atom_state, atom_mean])
        atom_state = layers.Dense(1)(atom_state)

        prediction = layers.GlobalAveragePooling1D()(atom_state)

        input_tensors = [atom_Input, bond_Input, connectivity_Input]
    else:
        global_state = layers.Dense(features_dim, activation='relu')(global_Input) 

        for i in range(num_messages):
            atom_state, bond_state, global_state = message_block(atom_state, bond_state,
                                                                 global_state, connectivity_Input, features_dim, i)

        prediction = layers.Dense(1)(global_state)

        input_tensors = [atom_Input, bond_Input, connectivity_Input, global_Input]

    model = tf.keras.Model(input_tensors, [prediction])
    model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(args.lr))

    model_path = "model_files/"+args.modelname+"/best_model.h5"

    checkpoint = ModelCheckpoint(model_path, monitor="val_loss",\
                                 verbose=2, save_best_only = True, mode='auto', period=1 )

    hist = model.fit(train_data,
                     validation_data=valid_data,
                     epochs=args.epoch,
                     verbose=2, callbacks = [checkpoint])

    model.load_weights(model_path)

    train_data_final = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset(train, preprocessor, args.sample_weight, False), output_signature=output_signature)\
        .cache()\
        .padded_batch(batch_size=args.batchsize)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    train_results = model.predict(train_data_final).squeeze()
    valid_results = model.predict(valid_data).squeeze()

    mae_train = np.abs(train_results - train['CN']).mean()
    mae_valid = np.abs(valid_results - valid['CN']).mean()

    if args.fold_number == -1:
        test_results = model.predict(test_data).squeeze()
        mae_test = np.abs(test_results - test['CN']).mean()
        print(len(train),len(valid),len(test))
        print(mae_train,mae_valid,mae_test)
    else:
        print("Fold number", args.fold_number)
        print(len(train),len(valid))
        print(mae_train,mae_valid)

        train['predicted'] = train_results
        valid['predicted'] = valid_results

        pd.concat([train, valid], ignore_index=True).to_csv('model_files/' + args.modelname +'kfold_'+str(args.fold_number)+'.csv',index=False)

    preprocessor.to_json("model_files/"+ args.modelname  +"/preprocessor.json")

def cnpred(smi):
    # best model name: 2_sw6
    model = tf.keras.models.load_model('model_files/2_sw6/best_model.h5', custom_objects = nfp.custom_objects)
    #print(model.summary())

    preprocessor = CustomPreprocessor(
        explicit_hs=False,
        atom_features=atom_features,
        bond_features=bond_features)
    preprocessor.from_json('model_files/2_sw6/preprocessor.json')

    inputs = preprocessor.construct_feature_matrices(smi)
    inputs = {key: np.expand_dims(inputs[key], axis=0) for key in ['atom','bond','connectivity','mol_features']}
    
    predicted_CN = model.predict( inputs   ).squeeze().squeeze()
    predicted_CN = np.round(predicted_CN,2)

    extractor = tf.keras.Model(model.inputs, [model.layers[1].output])
    features = np.array(extractor.predict(inputs).squeeze())

    ### Tanimoto similarity analysis ###
    df = pd.read_csv('data/CNdb_pred_results.csv')

    Similarities = []
    for _, row in df.iterrows():
        glob_vector_in_db = np.array([ float(x) for x in row['glob_vector'].split() ])

        C = np.sum( glob_vector_in_db * features  )
        A = np.sum( glob_vector_in_db * glob_vector_in_db   )
        B = np.sum( features * features   )

        S = C / ( A + B - C )

        Similarities.append(S)

    df['similarity'] = Similarities
    df = df.sort_values(by=['similarity'], ascending=False)

    if np.isclose(df.iloc[0].similarity, 1.0):
        df = df.iloc[1:]

    return predicted_CN, df.head(n=10)

def cnpred_df(df,args):
    model = tf.keras.models.load_model('model_files/'+ args.modelname +'/best_model.h5', custom_objects = nfp.custom_objects)
    preprocessor = CustomPreprocessor(
        explicit_hs=False,
        atom_features=atom_features,
        bond_features=bond_features)
    preprocessor.from_json('model_files/' + args.modelname +'/preprocessor.json')
    
    output_signature = (preprocessor.output_signature,
                        tf.TensorSpec(shape=(), dtype=tf.float32),
                        tf.TensorSpec(shape=(), dtype=tf.float32))

    df_data = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset(df, preprocessor, args.sample_weight, False), output_signature=output_signature)\
        .cache()\
        .padded_batch(batch_size=len(df))\
        .prefetch(tf.data.experimental.AUTOTUNE)

    pred_results = model.predict(df_data).squeeze()
    

    ### For SHAP ###
    import pickle
    weights_last_layer = [model.layers[-1].weights, model.layers[-3].weights, model.layers[-4].weights]
    with open('weights_last_layer.pkl','wb') as f:
        pickle.dump(weights_last_layer, f)  
    
    #extractor = tf.keras.Model(model.inputs, [model.layers[-6].output])
    extractor = tf.keras.Model(model.inputs, [model.layers[-5].input])
    last_layer_atom_features = extractor.predict(df_data)
    last_layer_atom_features = np.array(last_layer_atom_features)

    extractor = tf.keras.Model(model.inputs, [model.layers[-10].output])
    last_layer_global_features = extractor.predict(df_data)
    last_layer_global_features = np.array(last_layer_global_features)
    
    #print(last_layer_atom_features.shape)
    #print(last_layer_global_features.shape)

    features = [last_layer_atom_features, last_layer_global_features]
    with open('feat_vectors_for_shap.pkl','wb') as f:
        pickle.dump(features, f)

    
    #extractor = tf.keras.Model(model.inputs, [model.layers[-2].output])
    #extractor = tf.keras.Model(model.inputs, [model.layers[-2].input])
    #tmp = np.array(extractor.predict(df_data))
    #with open('tmp.pkl','wb') as f:
    #    pickle.dump(tmp, f)
    #############

    #### This part is for extracting feature (state) vectors (related to Fig. 5) ####
    extractor = tf.keras.Model(model.inputs, [model.layers[-1].input]) #global feature
    features = extractor.predict(df_data)

    '''
    #write atom features to csv
    extractor = tf.keras.Model(model.inputs, [model.layers[-6].output]) #atom features
    features = extractor.predict(df_data)
    from rdkit import Chem
    Num_atoms = [Chem.MolFromSmiles(smi).GetNumHeavyAtoms() for smi in df['Canonical_SMILES']]

    atom_features_list = []
    for i in range(len(features)):
        for j in range(Num_atoms[i]):
            atom_features_list.append([df.Canonical_SMILES.iloc[i],j, " ".join( [  "%.6f" % x  for x in features[i][j]])     ]  )

    pd.DataFrame(atom_features_list).to_csv('each_atom_feature_glob.csv', header = ['smiles','atom_index','atom_feature'])
    
    # averaged??
    avg_atom_features = []
    from rdkit import Chem
    Num_atoms = [Chem.MolFromSmiles(smi).GetNumHeavyAtoms() for smi in df['Canonical_SMILES']]

    for i in range(len(Num_atoms)):
        avg_atom_features_one_molecule = np.mean(features[i][0:Num_atoms[i]], axis=0)
        avg_atom_features.append(avg_atom_features_one_molecule)
    avg_atom_features = np.array(avg_atom_features)
    features = avg_atom_features
    '''

    df['predicted'] = pred_results

    features_str = []
    for feature in list(features):
        features_str.append(" ".join( [  "%.6f" % x  for x in feature]  ))

    print([layer.name for layer in model.layers])

    df['glob_vector'] = features_str
    new_df = df[['Canonical_SMILES','CN','predicted','glob_vector','Train/Valid/Test']]
    #new_df = df[['Canonical_SMILES','predicted','glob_vector']]
    #new_df.to_csv('prediction_results_avg_atom_feat.csv',index=False)
    new_df.to_csv('prediction_results.csv',index=False)
    ##################

if __name__ == '__main__':
    with tf.device(device):
        parser = ArgumentParser()
        parser.add_argument('-lr', type=float, default=1.0e-4, help='Learning rate (default=1.0e-4)')
        parser.add_argument('-batchsize', type=int, default=16, help='batch_size (default=16)')
        parser.add_argument('-epoch', type=int, default=1000, help='epoch (default=1000)')
        parser.add_argument('-layers', type=int, default=5, help='number of gnn layers (default=5)')
        parser.add_argument('-num_hidden', type=int, default=64, help='number of nodes in hidden layers (default=64)')

        parser.add_argument('-random_seed', type=int, default=1, help='random seed number used when splitting the dataset (default=1)')
        parser.add_argument('-split_option', type=int, default=2, help='8:1:1 split options - 0: just a random 8:1:1 split,\
                                                                                              1: Training set: Tier1,2,3, validation/test set: Tier 1 only,\
                                                                                              2: split from Tier 1 + split from Tier 2,3  (default=2)')

        parser.add_argument('-sample_weight', type=float, default=0.6, help='whether to use sample weights (default=0.6) If 1.0 -> no sample weights, if < 1.0 -> sample weights to Tier 2,3 methods')
        parser.add_argument('-tier1_only', action='store_true', default=False, help='whether to train the model using Tier 1 values only (default=False)')
        parser.add_argument('-no_glob_feat', action='store_true', default=False, help='If specified, no global features/updates are used (default=Use global features)')

        parser.add_argument('-predict', action="store_true", default=False, help='If specified, prediction is carried out (default=False)')
        parser.add_argument('-predict_df', action="store_true", default=False, help='If specified, prediction is carried out for molecules_to_predict.csv (default=False)')
        parser.add_argument('-smi', type=str, default='', help='SMILES for prediction')
        parser.add_argument('-modelname', type=str, default='', help='model name (default=blank)')
        parser.add_argument('-fold_number', type=int, default=-1, help='fold number for Kfold')
        args = parser.parse_args()

    if args.predict:
        predicted_CN, similarity_df = cnpred(args.smi)
        print(predicted_CN)
    elif args.predict_df:
        df = pd.read_csv('molecules_to_predict.csv')
        cnpred_df(df,args)
    else:
        main(args)
