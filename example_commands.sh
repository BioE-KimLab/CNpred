# Train MP-GNN with global features
python main.py -layers 5 -num_hidden 64 -batchsize 16 -split_option 2 -sample_weight 0.6 -modelname test
# Train MP-GNN without global features
python main.py -layers 5 -num_hidden 64 -batchsize 16 -split_option 2 -sample_weight 0.6 -modelname test2 -epoch 5 -no_glob_feat
# k-fold (fold_number: 0-9)
python main.py -layers 5 -num_hidden 64 -batchsize 16 -split_option 2 -sample_weight 0.6 -modelname test_fold0 -fold_number 0
python main.py -layers 5 -num_hidden 64 -batchsize 16 -split_option 2 -tier1_only -modelname test_fold0_tier1only -fold_number 0
# Predict CN of one compound (The best model is stored in model_files/2_sw6)
python main.py -predict -smi CCCC
# Predict CN of the dataframe (SMILES should be in the file 'molecules_to_predict.csv')
python main.py -predict_df -modelname 2_sw6 
