import os
from decouple import config
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn import model_selection

# INPUTPATH = config('INPUT')
# OUTPUTPATH = config('OUTPUT')

INPUTPATH = os.environ.get("INPUT")
OUTPUTPATH = os.environ.get("OUTPUT")

def cross_fold(data: str=None, K: int=None):
    """
    This function takes data and number of splits
    as an arguments and returns the fold of new data
    :param    data: raw dataframe input
    :param    K: number of splits used to create folds
    :return   df: pandas Kfolded dataframe"""
    outfile = ('fold.csv')
    outpath = OUTPUTPATH
    data['kfold'] = -1
    data = data.sample(frac=1).reset_index(drop=True) # randomize the rows of our dataframe
    num_bins = int(np.floor(1 + np.log2(len(data)))) # using Starge's rule to create bins
    data.loc[:, "bins"] = pd.cut(data["target"], bins=num_bins, labels=False) 
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=K)
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f
    data = data.drop("bins", axis=1)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    df = data.to_csv(f"{outpath}/{outfile}")
    return df
if __name__ == '__main__':
    load_dotenv()
    print(os.getenv('INPUTPATH'))
    df = pd.read_csv(INPUTPATH)
    #df = pd.read_csv(f"{INPUTPATH}/model.csv") 
    X = df.drop(columns=['estimated_stock_pct'])
    y = df['estimated_stock_pct']
    # create folds
    df = cross_fold(df, 10)