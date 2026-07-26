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


FONT_SIZE = 100
AMOUNT = 15000

def _make_dataset(repo, amount, font_size):
    import subprocess
    subprocess.run([
        "python", "word_on_background/generator.py",
        "--txt-file", "word_on_background/CollinsScrabbleWords(2019).txt",
        "--font-path", "word_on_background/fonts/dejavu-sans-mono.bold.ttf",
        "--save-dir", "dataset",
        "--font-size", str(font_size),
        "--amount", str(amount),
    ], cwd=repo, check=True)


@app.function(image=image, gpu="A10G", timeout=60 * 60 * 4, volumes={"/data": vol})
def pretrain_vqvae(amount: int = AMOUNT, epochs: int = 200, batch_size: int = 32):
    import subprocess
    repo = _clone()
    _make_dataset(repo, amount, FONT_SIZE)

    subprocess.run([
        "python", "pretrain.py",
        "--data-dir", "../../dataset/images",   
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--log-dir", "/data/runs/vqvae",        
        "--ckpt-dir", "/data/vqvae_ckpt",      
    ], cwd=f"{repo}/training/pretraining", check=True)
    vol.commit()


@app.function(image=image, gpu="A10G", timeout=60 * 60 * 4, volumes={"/data": vol})
def train_t5(amount: int = AMOUNT, epochs: int = 100, batch_size: int = 8):
    import os, shutil, subprocess
    repo = _clone()
    _make_dataset(repo, amount, FONT_SIZE)

    # put the pretrained VQVAE where load_models() actually looks for it
    ckpt_dir = f"{repo}/training/pretraining/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    shutil.copy("/data/vqvae_ckpt/best.pt", f"{ckpt_dir}/best.pt")

    subprocess.run(["python", "tokening.py"], cwd=f"{repo}/training", check=True)
    subprocess.run([
        "python", "train.py",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--mixed-precision",
        "--checkpoint-dir", "/data/t5_ckpt",
        "--log-dir", "/data/runs/t5",
    ], cwd=f"{repo}/training", check=True)
    vol.commit()