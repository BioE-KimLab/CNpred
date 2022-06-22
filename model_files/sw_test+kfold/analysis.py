
import glob
import pandas as pd
import numpy as np

for sw in range(1,11):
    valid_loss_eachfold = []
    valid_loss_all = 0.0
    for fold_num in range(10):
        df = pd.read_csv('./sw'+str(sw)+'_fold'+str(fold_num)+'/kfold_'+str(fold_num)+'.csv')
        valid = df[df['Train/Valid/Test'] == 'Valid']
        valid['AE'] = np.abs(valid.CN - valid.predicted)
        valid_loss_eachfold.append(valid['AE'].mean())
        valid_loss_all += np.sum(valid['AE'])
    valid_loss_all /= len(df)
    
    print(sw,np.round(valid_loss_all,2),[np.round(x,2) for x in valid_loss_eachfold], np.round(np.std(valid_loss_eachfold),2 ))
    #print(sw,valid_loss_all,np.std(valid_loss_eachfold)) 

