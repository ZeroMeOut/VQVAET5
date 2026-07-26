import os
import shutil
import argparse
import tempfile
import subprocess

VOLUME = "vqvaet5-data"

train_dir_name = "training/"
pretrain_dir_name = "training/pretraining/"

def parse_args():
    parser = argparse.ArgumentParser(description="Get modal data from the vqvaet5-data volume and create necessary folders")
    parser.add_argument("--train-dir", type=str, default=train_dir_name, help="Directory for training data")
    parser.add_argument("--pretrain-dir", type=str, default=pretrain_dir_name, help="Directory for pretraining data")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--only-get-pretrain-data", action="store_true", help="Only get pretraining data and skip training data")
    group.add_argument("--only-get-train-data", action="store_true", help="Only get training data and skip pretraining data")
    return parser.parse_args()


def _get(volume_path, dest):
    """Download a volume directory so its *files* land directly in dest.

    modal volume get copies the remote directory itself into the destination,
    so t5_ckpt/ -> training/checkpoints would give training/checkpoints/t5_ckpt/best.pt.
    inference.py loads training/checkpoints/best.pt, so stage through a temp
    directory and move the contents up a level.
    """
    with tempfile.TemporaryDirectory() as tmp:
        try:
            subprocess.run(["modal", "volume", "get", VOLUME, volume_path, tmp], check=True)
        except subprocess.CalledProcessError:
            exit(1)

        src = tmp
        nested = os.path.join(tmp, os.path.basename(volume_path.rstrip("/")))
        if os.path.isdir(nested):
            src = nested

        os.makedirs(dest, exist_ok=True)
        for name in os.listdir(src):
            target = os.path.join(dest, name)
            if os.path.isdir(target):
                shutil.rmtree(target)
            elif os.path.exists(target):
                os.remove(target)
            shutil.move(os.path.join(src, name), target)



def get_modal_data():
    args = parse_args()

    for d in (args.train_dir, args.pretrain_dir):
        os.makedirs(os.path.join(d, "runs"), exist_ok=True)
        os.makedirs(os.path.join(d, "checkpoints"), exist_ok=True)

    if not args.only_get_train_data:
        print("Getting pretraining data...")
        _get("runs/vqvae/", os.path.join(args.pretrain_dir, "runs"))
        _get("vqvae_ckpt/", os.path.join(args.pretrain_dir, "checkpoints"))

    if not args.only_get_pretrain_data:
        print("Getting training data...")
        _get("runs/t5/", os.path.join(args.train_dir, "runs"))
        _get("t5_ckpt/", os.path.join(args.train_dir, "checkpoints"))

if __name__ == "__main__":
    get_modal_data()
