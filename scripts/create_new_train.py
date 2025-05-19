import os
import sys

# Add the directory containing your module to the system path
sys.path.append('/home/rjsingh/causal_models')

import argparse
import glob
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tableshift import get_dataset



LOG_LEVEL = logging.DEBUG

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')


def load_data(experiment, cache_dir, 
    use_cached, split_name):

    dset = get_dataset(experiment, cache_dir, 
    use_cached=use_cached)

    # ----------------------- #
    # --- Get pandas data --- #
    # ----------------------- #

    file_pattern = os.path.join(dset.base_dir, '**', 
        f'{split_name}_*.csv')
    csv_files = glob.glob(file_pattern, recursive=True)

    df_list = []

    # Loop through each file and read it into a DataFrame
    for file in csv_files:
        df = pd.read_csv(file)
        df_list.append(df)

    # Concatenate all DataFrames in the list into a single DataFrame
    original_df = pd.concat(df_list, ignore_index=True)


    # --------------------- #
    # --- Sanity check ---- #
    # --------------------- #
    # make sure that raw imported data has the same dim
    # as what was used in the nastl paper
    
    X, _, _, _ = dset.get_pandas(split_name) 
    assert original_df.shape[0] == X.shape[0]
    del X

    return original_df, dset.base_dir



def get_oracle_test_split(experiment, cache_dir, 
    use_cached, rng):

    original_ts_df, base_dir = load_data(experiment, cache_dir, 
        use_cached, "ood_test")

    original_tr_df, _ = load_data(experiment, cache_dir, 
    use_cached, "train")

    # Get a random permutation of the DataFrame's indices
    shuffled_indices = rng.permutation(original_ts_df.index)

    # Calculate the split index
    split_index = min(int(len(original_ts_df) * 0.75), original_tr_df.shape[0])
    del original_tr_df

    # Split the indices into two groups
    indices_1 = shuffled_indices[:split_index]
    indices_2 = shuffled_indices[split_index:]

    # Create the two DataFrames based on the split indices
    oracle_df = original_ts_df.loc[indices_1].reset_index(drop=True)
    new_ts_df = original_ts_df.loc[indices_2].reset_index(drop=True)

    # save
    oracle_dir = os.path.join(base_dir, 'oracle', '1')
    os.makedirs(oracle_dir, exist_ok=True)
    oracle_df.to_csv(f'{oracle_dir}/oracle_00000.csv', index=False)

    new_ood_test_dir = os.path.join(base_dir, 
        'new_ood_test', '1')

    os.makedirs(new_ood_test_dir, exist_ok=True)
    new_ts_df.to_csv(f'{new_ood_test_dir}/new_ood_test_00000.csv', 
        index=False)

    return oracle_df.shape[0]

def get_new_train_split(experiment, cache_dir, 
    use_cached, n, random_seed):

    # --- Get matched size training data ----# 
    original_tr_df, base_dir = load_data(experiment, cache_dir, 
        use_cached, "train")

    # sample a fraction of this, make a script that only calls this, and then swap the names (make it double size of new_train)
    new_tr_df = original_tr_df.sample(n=n, 
        random_state= random_seed)

    new_id_train_dir = os.path.join(base_dir, 
        'sub_train', '1')

    os.makedirs(new_id_train_dir, exist_ok=True)
    new_tr_df.to_csv(f'{new_id_train_dir}/train_00000.csv', 
        index=False)



def main(experiment, cache_dir, use_cached, random_seed,
    debug: bool):
    if debug:
        print("[INFO] running in debug mode.")
        experiment = "_debug"

    rng = np.random.RandomState(random_seed)

    # n = get_oracle_test_split(experiment, cache_dir, 
    #     use_cached, rng)

   # get_new_train_split(experiment, cache_dir, 
    #    use_cached, 70000, random_seed)


    # --- final sanity check ---# 
    dset = get_dataset(experiment, cache_dir, 
        use_cached=use_cached)

    X_tr, y_tr, _, _ = dset.get_pandas('train') 
    print(X_tr.shape, y_tr.shape)

    

    X_or, y_or, _, _ = dset.get_pandas('oracle') 
    print(X_or.shape, y_or.shape)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp",
                        help="Directory to cache raw data files to.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Whether to run in debug mode. If True, various "
                             "truncations/simplifications are performed to "
                             "speed up experiment.")
    parser.add_argument("--use_cached", action="store_true", default=False,
                    help="use cached data?")

    parser.add_argument("--experiment", default="diabetes_readmission",
                        help="Experiment to run. Overridden when debug=True.")
    parser.add_argument("--random_seed", default=1234,
                    help="Random seed for replicability.")

    args = parser.parse_args()
    main(**vars(args))

