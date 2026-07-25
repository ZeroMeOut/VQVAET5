import modal

app = modal.App("vqvae-t5-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch", "torchvision", "transformers", "sentencepiece",
        "pandas", "pillow", "tensorboard",
    )
)

vol = modal.Volume.from_name("vqvaet5-data", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=60 * 60 * 4, volumes={"/data": vol})
def train(amount: int = 5000, epochs: int = 40, batch_size: int = 32):
    import os, shutil, subprocess

    repo = "/root/VQVAET5"
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/ZeroMeOut/VQVAET5.git", repo],
        check=True,
    )

    # the one thing that can't be regenerated — upload it once (see below)
    ckpt_dir = f"{repo}/training/encoder_pretraining/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    shutil.copy("/data/vqvae_best.pt", f"{ckpt_dir}/best.pt")

    # dataset is synthetic: render it here rather than shipping 5000 PNGs
    subprocess.run([
        "python", "word_on_background/generator.py",
        "--txt-file", "word_on_background/CollinsScrabbleWords(2019).txt",
        "--font-path", "word_on_background/fonts/dejavu-sans-mono.bold.ttf",
        "--save-dir", "dataset",
        "--amount", str(amount),
    ], cwd=repo, check=True)

    subprocess.run(["python", "tokening.py"], cwd=f"{repo}/training", check=True)

    subprocess.run([
        "python", "train.py",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--mixed-precision",
        "--checkpoint-dir", "/data/checkpoints",
        "--log-dir", "/data/runs",
    ], cwd=f"{repo}/training", check=True)

    vol.commit()