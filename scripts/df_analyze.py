import sys

# Add the directory containing your module to the system path
sys.path.append('/home/rjsingh/causal_models')
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
from tableshift.core import get_dataset
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
import os
import numpy as np


#acspubcov ORACLE
#df = pd.read_csv("/nfs/turbo/coe-rbg/rjsingh/results/acspubcov/20250504144622/tune_results_acspubcov_20250504144622_acspubcovdomain_split_varname_DISdomain_split_ood__label_group_dro.csv")
#print(df["ood_test_accuracy"]) # 16 is best
#acspubcov NEW_TRAIN: "/nfs/turbo/coe-rbg/rjsingh/results/acspubcov/20250504143925/tune_results_acspubcov_20250504143925_full.csv
# acspubcov TRAIN 7896178
#df = pd.read_csv("/nfs/turbo/coe-rbg/rjsingh/results/mimic_extract_mort_hosp/20250507142559/tune_results_mimic_extract_mort_hosp_20250507142559_full.csv")
#best = df["validation_accuracy"].idxmax()
#print(df['id_test_accuracy'][best])
#print(df["validation_accuracy"])
#print(df["ood_test_accuracy"])
#14
#print(df['ood_test_accuracy_conf'][16]) 
#brfss_diabetes node oracle: 7896418
# brfss_diabetes node train: slurm-7896211-4294967294.err
# brfss_diabetes node new_train: 7896215
#df = pd.read_csv("/nfs/turbo/coe-rbg/rjsingh/results/assistments/20250504212830/tune_results_assistments_20250504212830_assistmentsdomain_split_varname_school_iddomain_sp_ft_transformer.csv")
#print(df["validation_accuracy"])
#print(df["ood_test_accuracy"])
#print(df["ood_test_accuracy_conf"][19])
#print(df["new_ood_test_accuracy"]) 
#print(df['new_ood_test_accuracy_conf'][5]) 

# aldro train: 7896556
# aldro oracle: 7896557
# aldro new_train: 7896559

#physionet train: 7899461
#physionet new_train: 7899463
#physionet oracle: 7899464


# predictions = np.load("/nfs/turbo/coe-rbg/rjsingh/transformer_rest/new_train/2/ood_test/predictions.npz", allow_pickle = True)
# predictions = predictions["arr_0"].item() 
# print(predictions.keys())
# predictions_soft = predictions["predictions_soft"]
# predictions_hard = predictions["predictions_hard"]
# target = predictions["target"]
# input_data = predictions["input"]
# print(input_data.shape)
# print(predictions_hard.shape)
# print(predictions_soft.shape)
# print(target.shape)
# print(input_data[100])
# print(df["validation_accuracy"])
# print(df["ood_test_accuracy"])
# print(df["ood_test_accuracy_conf"])

#tune_results_brfss_blood_pressure_20250505113457_brfss_blood_pressuredomain_split_varname_BMI5CATdo_lightgbm.csv
# conf = False
# acc = 0.878811
# nobs = 6428
# if conf:
#     count = nobs * acc
#     acc_conf = proportion_confint(count, nobs, alpha=0.05, method="beta")
#     print(acc_conf)
# 2 sample proportion z test
cache_dir = '/nfs/turbo/coe-rbg/mmakar/tableshift/'
dset = get_dataset('assistments', cache_dir, use_cached=True)
X_tr, y_tr, _, _ = dset.get_pandas("ood_test")
id_len = y_tr.shape[0]
X_tr, y_tr, _, _ = dset.get_pandas("new_ood_test")
new_ood_len = y_tr.shape[0]
# id_test, new_ood_test
#count = np.array([31, 30])
nobs = np.array([id_len, new_ood_len])

# z-test
#stat, pval = proportions_ztest(count, nobs)

# comparing id_test and new_ood_test
# CI for difference in proportions

# NEW_TRAIN IID
prop1 = 0.59
# NEW_TRAIN OOD
prop2 = 0.63
diff = prop2 - prop1
se_diff = np.sqrt(prop1 * (1 - prop1) / nobs[0] + prop2 * (1 - prop2) / nobs[1])

# 95% CI
z = 1.96
ci_low = diff - z * se_diff
ci_upp = diff + z * se_diff
print(diff)
print(ci_low)
print(ci_upp)
"""
Physionet new_train
new_train
validation has length: 140287
id_test has length: 140288
train has length: 100801
ood_test has length: 33601
oracle
id_test has length: 140288
train has length: 100801
validation has length: 14934
ood_test has length: 33601
train
train has length: 1122299
validation has length: 140287
id_test has length: 140288
ood_test has length: 134402
ood_validation has length: 14934
new_ood_test has length: 33601
"""
# meps #7900790, train
# meps #7900794, new_train
# meps #7900850, oracle

# mimic new_train:7901688
# mimic oracle: 7901689