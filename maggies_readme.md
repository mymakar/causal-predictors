
## step 0: setup conda environment
-- conda create -n cp python=3.11
-- conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
-- conda install nvidia/label/cuda-11.8.0::cuda-toolkit
-- pip3 install -U "ray[data,train,tune,serve]"
-- pip3 install xport fairlearn frozendict datasets folktables rtdl tab-transformer-pytorch optuna catboost
-- export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nfs/turbo/coe-rbg/mmakar/miniconda/envs/cp/lib/
-- python -m pip install statsmodels
-- pip install xgboost
-- pip install lightgbm
-- pip install category_encoders
-- pip install hyperopt


### Step 0 alt: set up older conda env 
-- conda create -n cp2 python=3.10
-- conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
-- conda install nvidia/label/cuda-11.8.0::cuda-toolkit
-- pip install ray==2.2.0
-- pip3 install xport fairlearn frozendict datasets folktables rtdl tab-transformer-pytorch optuna catboost
-- export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nfs/turbo/coe-rbg/mmakar/miniconda/envs/cp2/lib/
-- python -m pip install statsmodels
-- pip install xgboost lightgbm category_encoders hyperopt
-- pip install ray[tune]==2.2.0
-- pip install lightgbm-ray==0.1.8
-- pip uninstall ray 
-- conda install conda-forge::ray-all=2.2.0
-- pip uninstall pyarrow 
-- pip install pyarrow==11.0.0
-- pip uninstall datasets 
-- pip install datasets==2.11.0
-- conda install conda-forge::ray-tune=2.2.0
-- pip uninstall pydantic 
-- pip install pydantic==1.10.2

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
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nfs/turbo/coe-rbg/mmakar/miniconda/envs/cp/lib/
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
    --experiment diabetes_readmission \
    --use_cached \
    --cache_dir  /nfs/turbo/coe-rbg/mmakar/tableshift/ 
```


## step 3: run models 

### generic run
python experiments_causal/run_experiment.py  \
    --experiment diabetes_readmission \
    --model lightgbm \
    --cache_dir  /nfs/turbo/coe-rbg/mmakar/tableshift/ \
    --save_dir /nfs/turbo/coe-rbg/mmakar/causalfeatures/results \
    --use_cached \
    --split_mode train \


### lightgbm and xgb
python examples/run_expt_headroom.py --experiment acspubcov --cache_dir  /nfs/turbo/coe-rbg/mmakar/tableshift/ --use_cached --model label

### catboost (college_scorecard, mimic_extract_los_3)
python scripts/train_catboost_optuna_headroom.py \
    --experiment college_scorecard \
    --cache_dir  /nfs/turbo/coe-rbg/mmakar/tableshift/ \
    --use_cached \
    --use_gpu \
    --split_mode new_train 

### Ray headroom
python scripts/ray_train_headroom.py \
    --models label_group_dro \
    --experiment acspubcov \
    --cache_dir  /nfs/turbo/coe-rbg/mmakar/tableshift/ \
    --use_cached \
    --num_samples 2\
    --max_concurrent_trials 2

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


-- conda install nvidia/label/cuda-11.8.0::cuda-toolkit
-- pip3 install -U "ray[data,train,tune,serve]"
-- pip3 install xport fairlearn frozendict datasets folktables rtdl tab-transformer-pytorch optuna catboost
-- export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nfs/turbo/coe-rbg/mmakar/miniconda/envs/cp/lib/
-- python -m pip install statsmodels
-- pip install xgboost
-- pip install lightgbm
-- pip install category_encoders

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
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nfs/turbo/coe-rbg/mmakar/miniconda/envs/cp/lib/
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

### generic run
python experiments_causal/run_experiment.py \
    --experiment college_scorecard \
    --model catboost \
    --cache_dir  /nfs/turbo/coe-rbg/mmakar/tableshift/  \
    --save_dir /nfs/turbo/coe-rbg/mmakar/causalfeatures/results \
    --use_cached 

### lightgbm and xgb
python examples/run_expt_headroom.py --experiment acspubcov --cache_dir  /nfs/turbo/coe-rbg/mmakar/tableshift/ --use_cached --model label

### catboost (college_scorecard, mimic_extract_los_3)
python scripts/train_catboost_optuna_headroom.py \
    --experiment college_scorecard \
    --cache_dir  /nfs/turbo/coe-rbg/mmakar/tableshift/ \
    --use_cached \
    --use_gpu \
    --split_mode new_train 

### ray training 
python scripts/ray_train_headroom.py \
    --models label_group_dro \
    --experiment acspubcov \
    --cache_dir  /nfs/turbo/coe-rbg/mmakar/tableshift/ \
    --use_cached \
    --num_samples 1\
    --max_concurrent_trials 1



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
- tableshift/models/compat.py
- tableshift/models/expgrad.py
- tableshift/models/fastdro/datasets.py
- tableshift/models/ray_utils.py
- tableshift/models/training.py
- tableshift/models/training_headroom.py

## Edits when going from cp to cp2: 
-- /nfs/turbo/coe-rbg/mmakar/miniconda/envs/cp2/lib/python3.10/site-packages/ray/air/util/tensor_extensions/arrow.py
line 4: 
from pip._vendor.packaging.version import parse as parse_version
-- /nfs/turbo/coe-rbg/mmakar/miniconda/envs/cp2/lib/python3.10/site-packages/ray/data/_internal/util.py
line 57: 
from pip._vendor.packaging.version import parse as parse_version
-- tableshift/models/expgrad.py 
line 12 from ray.air.checkpoint import Checkpoint
-- tableshift/models/ray_utils.py 
line 18 from ray.train.torch import TorchCheckpoint, TorchTrainer
(I believe this was commented out earlier)
-- /home/mmakar/projects/causal-predictors/tableshift/models/compat.py
line 15 from ray.air.checkpoint import Checkpoint
