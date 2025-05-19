"""
Usage:

python scripts/train_catboost_optuna \
	--experiment adult \
	--use_gpu \
	--use_cached

"""
import sys

# Add the directory containing your module to the system path
sys.path.append('/path/to/project')



import argparse
import logging
import os

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, \
	average_precision_score
from sklearn.utils import resample

from tableshift.core import get_dataset
from tableshift.core.utils import timestamp_as_int

from experiments_causal.metrics import balanced_accuracy_score
from experiments_causal.run_experiment import bootstrap_auroc
from statsmodels.stats.proportion import proportion_confint

def bootstrap_auroc(y_true, y_pred, n_bootstraps=100, ci=95, random_seed=42):
	np.random.seed(random_seed)
	bootstrapped_scores = []
	y_true = y_true.to_numpy()
	for _ in range(n_bootstraps):
		# Resample with replacement
		indices = resample(np.arange(len(y_true)), replace=True)
		if len(np.unique(y_true[indices])) < 2:
			# Skip this bootstrap sample if only one class is present
			continue
		score = roc_auc_score(y_true[indices], y_pred[indices])
		bootstrapped_scores.append(score)

	# Compute confidence intervals
	lower_percentile = (100 - ci) / 2
	upper_percentile = 100 - lower_percentile
	ci_lower = np.percentile(bootstrapped_scores, lower_percentile)
	ci_upper = np.percentile(bootstrapped_scores, upper_percentile)

	return ci_lower, ci_upper


def evaluate(model: CatBoostClassifier, X: pd.DataFrame, y: pd.Series,
			 split: str, split_mode: str) -> dict:
	yhat_hard = model.predict(X)
	yhat_soft = model.predict_proba(X)[:, 1]
	metrics = {}
	metrics[f"{split}_accuracy"] = accuracy_score(y, yhat_hard)
	metrics[f"{split}_auc"] = roc_auc_score(y, yhat_soft)
	metrics[f"{split}_map"] = average_precision_score(y, yhat_soft)

	metrics[f"{split}_num_samples"] = len(y)
	metrics[f"{split}_ymean"] = np.mean(y).item()
	"""
	Add eval for ood_test and new_ood_test here
	"""
	if split == 'ood_test' or split == 'new_ood_test':
			nobs = len(y)
			count = nobs * metrics[f"{split}_accuracy"] 
			acc_conf = proportion_confint(count, nobs, 
				alpha=0.05, method="beta")
			metrics[split + "_accuracy_conf_lower"] = acc_conf[0]
			metrics[split + "_accuracy_conf_upper"] = acc_conf[1]
			auc = metrics[f"{split}_auc"]
			auc_lower, auc_upper = bootstrap_auroc(y, yhat_soft)
			metrics[split + "_auc_conf_lower"] = auc_lower
			metrics[split + "_auc_conf_upper"] = auc_upper 
			balanced_acc, balanced_acc_se = balanced_accuracy_score(
					target=y, prediction=yhat_hard
			)
			metrics[split + "_balanced"] = balanced_acc
			balanced_acc_conf = (
					balanced_acc - 1.96 * balanced_acc_se,
					balanced_acc + 1.96 * balanced_acc_se,
			)
			metrics[split + "_balanced" + "_conf_lower"] = balanced_acc_conf[0]
			metrics[split + "_balanced" + "_conf_upper"] = balanced_acc_conf[1]

	return metrics, yhat_hard, yhat_soft


def main(experiment: str, cache_dir: str, results_dir: str, num_samples: int,
		 random_seed:int, 
		 use_gpu: bool, use_cached: bool, split_mode: str):

	if split_mode not in ['train', 'new_train', 'oracle']: 
		raise NotImplementedError(
			'Choose from train, new_train and oracle')
	start_time = timestamp_as_int()

	dset = get_dataset(experiment, cache_dir, use_cached=use_cached)
	uid = dset.uid

	if split_mode == 'train':
		X_tr, y_tr, _, _ = dset.get_pandas("train")
		X_val, y_val, _, _ = dset.get_pandas("validation")
	elif split_mode == 'new_train': 
		X_tr, y_tr, _, _ = dset.get_pandas("new_train")
		X_val, y_val, _, _ = dset.get_pandas("validation")   
	elif split_mode == 'oracle': 
		X_tr, y_tr, _, _ = dset.get_pandas("oracle")
		X_val, y_val, _, _ = dset.get_pandas("ood_validation")           

	def optimize_hp(trial: optuna.trial.Trial):
		cb_params = {
			# Same tuning grid as https://arxiv.org/abs/2106.11959,
			# see supplementary section F.4.
			'learning_rate': trial.suggest_float('learning_rate', 1e-3,
												 1., log=True),
			'depth': trial.suggest_int('depth', 3, 10),
			'bagging_temperature': trial.suggest_float(
				'bagging_temperature', 1e-6, 1., log=True),
			'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 100, log=True),
			'leaf_estimation_iterations': trial.suggest_int(
				'leaf_estimation_iterations', 1, 10),

			"use_best_model": True,
			"task_type": "GPU" if use_gpu else "CPU",
			'random_seed': random_seed,
		}

		model = CatBoostClassifier(**cb_params)
		model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
		y_pred = model.predict(X_val)
		return accuracy_score(y_val, y_pred)

	study = optuna.create_study(direction="maximize")
	study.optimize(optimize_hp, n_trials=num_samples)
	# study.optimize(optimize_hp, n_trials=2)
	print('Trials:', len(study.trials))
	print('Best parameters:', study.best_trial.params)
	print('Best score:', study.best_value)
	print("training completed! retraining model with best params and "
			"evaluating it.")

	clf_with_best_params = CatBoostClassifier(**study.best_trial.params)
	clf_with_best_params = clf_with_best_params.fit(X_tr, y_tr)

	expt_results_dir = os.path.join(results_dir, experiment, str(start_time))

	metrics = {}
	model_name = "catboost"
	metrics["estimator"] = model_name
	metrics["domain_split_varname"] = dset.domain_split_varname
	metrics["domain_split_ood_values"] = str(dset.get_domains("ood_test"))
	metrics["domain_split_id_values"] = str(dset.get_domains("id_test"))

	splits = (
		'id_test', 'ood_test', 'new_ood_test', 'ood_validation', 'validation') if dset.is_domain_split else (
		'test', 'validation')
	for split in splits:
		X, y, _, _ = dset.get_pandas(split)
		_metrics, yhat_hard, yhat_soft = evaluate(
			clf_with_best_params, X, y, split, split_mode)
		print(_metrics)
		metrics.update(_metrics)
		if split == 'new_ood_test': 
			yhat_hard_keep = yhat_hard.copy()
			yhat_soft_keep = yhat_soft.copy()
			y_keep = y.copy()

	iter_fp = os.path.join(
		expt_results_dir,
		f"tune_results_{experiment}_{start_time}_{uid[:100]}_"
		f"{model_name}_{split_mode}.csv")
	iter_preds = os.path.join(
		expt_results_dir,
		f"prediction_results_{experiment}_{start_time}_{uid[:100]}_"
		f"{model_name}_{split_mode}.csv")

	if not os.path.exists(expt_results_dir):
		os.makedirs(expt_results_dir)
	logging.info(f"writing results for {model_name} to {iter_fp}")
	pd.DataFrame(metrics, index=[1]).to_csv(iter_fp, index=False)
	res_df = pd.DataFrame({
		'y': y_keep,
		'hard_pred': yhat_hard_keep, 
		'soft_pred': yhat_soft_keep
		})
	res_df.to_csv(iter_preds, index=False)
	print(res_df.head())



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--cache_dir", default="tmp",
						help="Directory to cache raw data files to.")
	parser.add_argument("--experiment", default="adult",
						help="Experiment to run. Overridden when debug=True.")
	parser.add_argument("--num_samples", type=int, default=100,
						help="Number of hparam samples to take in tuning "
							 "sweep.")
	parser.add_argument("--results_dir", default="./optuna_results",
						help="where to write results. CSVs will be written to "
							 "experiment-specific subdirectories within this "
							 "directory.")
	parser.add_argument("--random_seed", default=42, type=int)
	parser.add_argument("--use_cached", default=False, action="store_true",
						help="whether to use cached data.")
	parser.add_argument("--use_gpu", action="store_true", default=False,
						help="whether to use GPU (if available)")
	parser.add_argument("--split_mode", default="train",
					help="train, new_train or oracle.")

	args = parser.parse_args()
	main(**vars(args))