import os
import os.path as path
import json
import argparse
import itertools

parser = argparse.ArgumentParser(
    description="Prepare experimental environment for cross validation (CV)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--cv_dir", default="cross_validation/",
    help="a folder where CV will be performed"
)
parser.add_argument(
    "--n_folds", type=int, default=5,
    help="number of CV folds"
)
parser.add_argument(
    "--code_dir", default="code/",
    help="a folder that contains all the train/test code"
)
parser.add_argument(
    "--data_dir", default="data/",
    help="a folder that contains all the experimental data" 
)
args = parser.parse_args()

# Get file and volume list
train_dir = path.join(args.data_dir, "train")
train_files = [
    path.join(train_dir, f) for f in os.listdir(train_dir)
    if "new" in f.split("_")[1] == "new"
]
test_dir = path.join(args.data_dir, "test")
test_files = [
    path.join(test_dir, f) for f in os.listdir(test_dir)
    if f.split("_")[1] == "new"
]
files = train_files + test_files
volume_names = sorted(list(set(
    ["_".join(path.split(f)[-1].split("_")[:2]) for f in files]
)))

# Get fold lists
fold_size = len(volume_names) / args.n_folds
fold_lists = [
    volume_names[fold_id * fold_size:(fold_id + 1) * fold_size]
    for fold_id in range(args.n_folds)
]

# Prepare fold folder
for fold_id, fold_list in enumerate(fold_lists):
    print("Preparing fold {}".format(fold_id))

    fold_dir = path.join(args.cv_dir, "fold{}".format(fold_id))
    if not path.isdir(fold_dir):
        os.makedirs(fold_dir)

    # Prepare code folder
    dst_code_dir = path.join(fold_dir, "code")
    src_code_dir = path.abspath(args.code_dir)
    if path.islink(dst_code_dir):
        os.remove(dst_code_dir)
    os.symlink(src_code_dir, dst_code_dir)

    # Prepare data folder
    dst_data_dir = path.join(fold_dir, "data")
    if not path.isdir(dst_data_dir):
        os.makedirs(dst_data_dir)

    for phase in ["train", "test"]:
        if phase == "test":
            phase_list = fold_list
        else:
            phase_list = list(itertools.chain.from_iterable(
                fold_lists[:fold_id] + fold_lists[fold_id+1:]
            ))
        dst_phase_dir = path.join(dst_data_dir, phase)
        if not path.isdir(dst_phase_dir):
            os.makedirs(dst_phase_dir)

        # Save phase file
        phase_list_file = path.join(dst_data_dir, "{}.json".format(phase))
        with open(phase_list_file, "w") as f:
            json.dump(phase_list, f, indent=4, sort_keys=True)

        # Create soft links to the phase files
        for src_file in files:
            file_name = path.split(src_file)[-1]
            volume_name = "_".join(file_name.split("_")[:2])
            if volume_name not in phase_list:
                continue

            src_file = path.abspath(src_file)
            dst_file = path.join(dst_phase_dir, file_name)
            if path.islink(dst_file):
                os.remove(dst_file)
            os.symlink(src_file, dst_file)
            assert os.path.exists(dst_file)