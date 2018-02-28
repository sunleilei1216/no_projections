import os
import os.path as path
import json
import argparse

parser = argparse.ArgumentParser(
    description="Collect the cross-validation results",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--cv_dir", default="cross_validation/",
    help="the cross validation folder"
)
parser.add_argument(
    "--results_dir", default="results/",
    help="a folder that stores the collected cross-validation results"
)
parser.add_argument(
    "--model_names", nargs="+",
    default=['2d_no_clips_24'],
    help="the name of the models from which the results will be collected"
)
args = parser.parse_args()

fold_names = [
    d for d in os.listdir(args.cv_dir) if path.isdir(path.join(args.cv_dir, d))
]

for model_index, model_name in enumerate(args.model_names):
    contour_scores = []
    dice_scores = []
    for fold_name in fold_names:
        # Load scores
        model_dir = path.join(
            args.cv_dir, fold_name, "results", model_name
        )
        fold_contour_score_file = path.join(model_dir, "contour_scores.json")
        fold_dice_score_file = path.join(model_dir, "dice_scores.json")
        with open(fold_contour_score_file) as f:
            fold_contour_scores = json.load(f)
        with open(fold_dice_score_file) as f:
            fold_dice_scores = json.load(f)

        # Aggregate scores
        for scores, fold_scores in [
            (contour_scores, fold_contour_scores),
            (dice_scores, fold_dice_scores),
        ]:
            if len(scores) == 0:
                scores += fold_scores
            else:
                for t in scores[0]:
                    scores[0][t] += fold_scores[0][t]
                for t in scores[1]:
                    for i in range(len(scores[1][t])):
                        scores[1][t][i] += fold_scores[1][t][i]
    # Average scores
    for scores in (contour_scores, dice_scores):
        for t in scores[0]:
            scores[0][t] /= len(fold_names)
        for t in scores[1]:
            for i in range(len(scores[1][t])):
                scores[1][t][i] /= len(fold_names)

    # Make results directory
    cv_results_dir = path.join(
        args.results_dir, model_name, "cross_validation"
    )
    if not path.isdir(cv_results_dir):
        os.makedirs(cv_results_dir)

    # Save cv results
    contour_scores_file = path.join(cv_results_dir, "contour_scores.json")
    dice_scores_file = path.join(cv_results_dir, "dice_scores.json")
    with open(contour_scores_file, "w") as f:
        json.dump(contour_scores, f, indent=4, sort_keys=True)
    with open(dice_scores_file, "w") as f:
        json.dump(dice_scores, f, indent=4, sort_keys=True)
