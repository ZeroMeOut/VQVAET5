## Notes
Still work in progress. The goal is to create a dount style model (see [here](https://arxiv.org/pdf/2111.15664) for context) for extracting items bought in a receipt. Atm I plan on using VQVAE as the encoder and Google's T5 model as the decoder, tho it may change in the future. Oh and the dataset is purely synthetic, because I can generate a ton of samples that look like receipts and have control over the format. So it mayyyy not work on real receipts but it could be finetuned to.

## Timeline notes
- I have trained the vqvae model but the quality is not that great imo, I will try to see what I can do
- I am running out of memeory while trying to train
- Fixed it :)
- The training of the VQVAET5 is very slow for 5000 images per epoch and my ass doesn't want to rent a GPU, I may have reduce it.