import pandas as pd

df = pd.read_csv("/nfs/turbo/coe-rbg/rjsingh/results/acspubcov/20250502152612/tune_results_acspubcov_20250502152612_acspubcovdomain_split_varname_DISdomain_split_ood__label_group_dro.csv")
print(df.columns) 
print(df["id_test_accuracy"])