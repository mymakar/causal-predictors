"""Python script to run experiment and record the performance."""
import sys
# Add the directory containing your module to the system path
sys.path.append('/home/mmakar/projects/causal-predictors')

import argparse
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import resample
import json
from statsmodels.stats.proportion import proportion_confint

from tableshift import get_dataset
from tableshift.models.training_headroom import train
from tableshift.models.utils import get_estimator
from tableshift.models.default_hparams import get_default_config
from experiments_causal.metrics import balanced_accuracy_score

def bootstrap_auroc(y_true, y_pred, n_bootstraps=100, ci=95, random_seed=42):
    np.random.seed(random_seed)
    bootstrapped_scores = []

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

def main(
        experiment: str,
        model: str,
        cache_dir: str,
        save_dir: str,
        use_cached: bool,
        split_mode: str,  
        debug: bool):
    """Run the experiment with the specified model.

    Parameters
    ----------
    experiment : str
        The name of the experiment to run.
    model : str
        The name of the model to train.
    cache_dir : str
        Directory to cache raw data files to.
    save_dir : str
        Directory to save result files to.
    use_cached: bool, 
        Use cached dataset?
    split_mode: 
        train on original train, new train or
        oracle 
    debug : bool
        Debug mode.

    Returns
    -------
    None.

    """
    cache_dir = Path(cache_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=False)

    dset = get_dataset(experiment, cache_dir, 
        use_cached=use_cached)
    # manually add the names of the new splits
    dset.splits = dset.splits + ["new_ood_test", "oracle", "new_train"]
    if debug:
        print("[INFO] running in debug mode.")
        experiment = "_debug"

    config = get_default_config(model, dset)
    estimator = get_estimator(model, **config)
    estimator = train(estimator, dset, split_mode=split_mode, 
        config=config)

    if not isinstance(estimator, torch.nn.Module):
        evaluation = {}
        # Case: non-pytorch estimator; perform test-split evaluation.
        test_splits = (
            ["id_test", "ood_test", "validation", "ood_validation", 
                "new_ood_test"] if dset.is_domain_split else ["test"]
        )

        for test_split in test_splits:
            # Fetch predictions and labels for a sklearn model.
            X_te, y_te, _, _ = dset.get_pandas(test_split)
            predictor_names = list(X_te.columns)
            X_te = X_te.astype(float)
            yhat_hard_te = estimator.predict(X_te)
            yhat_soft_te = estimator.predict_proba(X_te)[:,1]

            # save individual predictions
            if test_split == 'new_test_ood':
                pred_df = pd.DataFrame({'y': y_te, 'yhat_hard': yhat_hard_te, 
                    'yhat_soft': yhat_soft_te})
                print(pred_df.head())
                pred_file_name = f"{str(SAVE_DIR_EXP)}/{experiment}_{split_mode}_{model}_pred.csv" 
                pred_df.to_csv(pred_file_name, index=False)

            # Calculate accuracy
            acc = accuracy_score(y_true=y_te, y_pred=yhat_hard_te)
            evaluation[f'{test_split}_accuracy'] = acc
            nobs = len(y_te)
            count = nobs * acc
            # beta : Clopper-Pearson interval based on Beta distribution
            acc_conf = proportion_confint(count, nobs, alpha=0.05, method="beta")
            evaluation[test_split + "accuracy_conf"] = acc_conf

            # calculate auc
            evaluation[f"{test_split}_auc"] = roc_auc_score(y_te, yhat_soft_te)

            # bootstrap to get the confidence interval: 
            auc_lower, auc_upper = bootstrap_auroc(y_te.values, yhat_soft_te)
            evaluation[test_split + "auc_conf_lower"] = auc_lower
            evaluation[test_split + "auc_conf_upper"] = auc_upper 

            print(f"training completed! {test_split} accuracy: {acc:.4f}")

            # Calculate balanced accuracy
            balanced_acc, balanced_acc_se = balanced_accuracy_score(
                target=y_te, prediction=yhat_hard_te
            )
            evaluation[test_split + "_balanced"] = balanced_acc
            balanced_acc_conf = (
                balanced_acc - 1.96 * balanced_acc_se,
                balanced_acc + 1.96 * balanced_acc_se,
            )
            evaluation[test_split + "_balanced" + "_conf"] = balanced_acc_conf
            print(
                f"training completed! {test_split} balanced accuracy: {balanced_acc:.4f}"
            )

        # Open a file in write mode
        SAVE_DIR_EXP = save_dir / experiment
        SAVE_DIR_EXP.mkdir(exist_ok=True)
        json_file_name = f"{str(SAVE_DIR_EXP)}/{experiment}_{split_mode}_{model}_eval.json" 
        with open(json_file_name, "w") as f:
            # Use json.dump to write the dictionary into the file
            # This original line throws an error
            # evaluation["features"] = dset.predictors
            evaluation["features"] = predictor_names
            json.dump(evaluation, f)

    else:
        # Case: pytorch estimator; eval is already performed + printed by train().
        print("training completed!")
        evaluation = estimator.fit_metrics
        evaluation_balanced = estimator.fit_metrics_balanced
        for test_split in ["id_test", "ood_test", "validation", "ood_validation", 
                "new_test_ood"]:
            # Get accuracy
            # Fetch predictions and labels for a sklearn model.
            X_te, y_te, _, _ = dset.get_pandas(test_split)
            X_te = X_te.astype(float)
            nobs = len(y_te)
            acc = evaluation[test_split]
            count = nobs * acc
            acc_conf = proportion_confint(count, nobs, alpha=0.05, method="beta")
            evaluation[test_split + "_conf"] = acc_conf

            # Get balanced accuracy
            balanced_acc = evaluation_balanced["score"][test_split]
            balanced_acc_se = evaluation_balanced["se"][test_split]
            evaluation[test_split + "_balanced"] = balanced_acc
            balanced_acc_conf = (
                balanced_acc - 1.96 * balanced_acc_se,
                balanced_acc + 1.96 * balanced_acc_se,
            )
            evaluation[test_split + "_balanced" + "_conf"] = balanced_acc_conf
            print(
                f"training completed! {test_split} accuracy: {evaluation[test_split]:.4f}"
            )
            print(
                f"training completed! {test_split} balanced accuracy: {balanced_acc:.4f}"
            )

        # Open a file in write mode
        SAVE_DIR_EXP = save_dir / experiment
        SAVE_DIR_EXP.mkdir(exist_ok=True)
        with open(f"{str(SAVE_DIR_EXP)}/{model}_eval.json", "w") as f:
            evaluation["features"] = dset.predictors
            json.dump(evaluation, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir", default="tmp", help="Directory to cache raw data files to."
    )
    parser.add_argument(
        "--save_dir", default="tmp/results", help="Directory to save result files to."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Whether to run in debug mode. If True, various "
        "truncations/simplifications are performed to "
        "speed up experiment.",
    )
    parser.add_argument(
        "--experiment",
        default="diabetes_readmission",
        help="Experiment to run. Overridden when debug=True.",
    )
    parser.add_argument("--use_cached", default=False, action="store_true",
                        help="whether to use cached data.")

    parser.add_argument("--model", default="histgbm", help="model to use.")
    parser.add_argument("--split_mode", default="train",
        help="train, new_train or oracle.")

    args = parser.parse_args()
    main(**vars(args))
