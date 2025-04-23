## step 0: setup conda environment
- Run `conda env create -f environment.yml` where `environment.yml` is the file provided by the causal predictors github
- This gave me a ton of "errors". I went ahead with the install anyway. Think those errors are because the env.yml file specified an older version of python..
- After that I had to install the following manually: 
-- conda create -n cp python=3.11
-- conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
-- conda install nvidia/label/cuda-11.8.0::cuda-toolkit
-- pip3 install -U "ray[data,train,tune,serve]"
-- pip3 install xport fairlearn frozendict datasets folktables rtdl tab-transformer-pytorch optuna catboost
-- export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nfs/turbo/coe-rbg/mmakar/miniconda/envs/causalfeatures/lib/



### Notes
- Remember to run: 
```
salloc --cpus-per-task=1 --ntasks-per-node=1 --account=precisionhealth_owned1 --partition=precisionhealth --time=2:00:00 --tasks-per-node=1 --mem=500gb --gres=gpu:1

conda activate cp
```
- If you get an error: 
```
 version `GLIBCXX_3.4.26' not found (required by /nfs/turbo/coe-rbg/mmakar/miniconda/envs/causalfeatures/lib/python3.8/site-packages/scipy/linalg/_matfuncs_sqrtm_triu.cpython-38-x86_64-linux-gnu.so)
```
run
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nfs/turbo/coe-rbg/mmakar/miniconda/envs/causalfeatures/lib/
```

## step 1: create datasets: 
This step creates the original data splits. One note about this: it saves the data to directories with *really* weird names but I couldn't find an easy way to change that. So it is what it is. 
```
python scripts/cache_task.py \
    --experiment college_scorecard \
    --cache_dir /nfs/turbo/coe-rbg/mmakar/tableshift/
```
## step 2: create new data splits 
```
python scripts/create_new_splits.py \
    --experiment college_scorecard \
    --use_cached \
    --cache_dir  /nfs/turbo/coe-rbg/mmakar/tableshift/ 
```


## step 3: run models 
### lightgbm and xgb
python examples/run_expt_headroom.py --experiment acspubcov --cache_dir  /nfs/turbo/coe-rbg/mmakar/tableshift/ --use_cached --model label

### catboost (college_scorecard, mimic_extract_los_3)
python scripts/train_catboost_optuna_headroom.py \
    --experiment college_scorecard \
    --cache_dir  /nfs/turbo/coe-rbg/mmakar/tableshift/ \
    --use_cached \
    --use_gpu \
    --split_mode new_train 

### not used yet 
python scripts/ray_train_headroom.py \
    --models lightgbm \
    --experiment brfss_blood_pressure \
    --cache_dir  /nfs/turbo/coe-rbg/mmakar/tableshift/ \
    --use_cached



git 
## note these modifications: 
/home/mmakar/projects/causal-predictors/tableshift/configs/hparams.py
https://github.com/mlfoundations/tableshift/pull/4
[added the import sys] /home/mmakar/projects/causal-predictors/scripts/train_catboost_optuna.py
[added the import sys] /home/mmakar/projects/causal-predictors/scripts/cache_task.py
tableshift/core/tabular_datasets.py

### list of all the modified files 
- environment.yml
- examples/run_expt.py
- scripts/cache_task.py
- scripts/train_catboost_optuna.py
- tableshift/configs/hparams.py
- tableshift/core/data_source.py
- tableshift/core/tabular_dataset.py
- tableshift/models/ray_utils.py
- tableshift/models/training.py

