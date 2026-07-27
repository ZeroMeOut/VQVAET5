
# VQVAET5

A smol VLM that I trained to extract words from a white background (initally it was from receipts but I switched objective midway)


## Getting Started
Run the following in order (I assume you have uv installed)

```bash
git clone "https://github.com/ZeroMeOut/VQVAET5.git"
cd VQVAET5
uv sync
python inference.py
```
If you want to get an evaluation of how good the model is, you can run `python inference.py --eval` and you should get something like this in your terminal:

``` 
Loading weights: 100%|█████████████████████████████████████████| 131/131 [00:00<00:00, 18736.70it/s]

Scoring 500 holdout words (batch size 4)...
  500/500

  exact match:          435/500 (87.0%)
  character error rate: 2.2%
  mean edit distance:   0.20 chars per word

  sample errors:
    target='THEOLOGISATIONS'    predicted='HEEOLOGISATION'     distance=3
    target='OFTENEST'           predicted='OFFTENEST'          distance=1
    target='BRRR'               predicted='BRR'                distance=1
    target='UNWORLDLIEST'       predicted='UNWORDLLIEST'       distance=2
    target='NARWHAL'            predicted='NAWRHAL'            distance=2
    target='VITRIOLIZES'        predicted='VIVRIOLIZES'        distance=1
    target='BETHWACK'           predicted='BETWACK'            distance=1
    target='OVERREPRESENTED'    predicted='VOVEREPRESSENED'    distance=4
    target='BISECTRICES'        predicted='BISICTRICES'        distance=1
```
Yes, that is what I am getting rn from my trained model on the default args of the images :)

There are some specific args that will make the model's output to be bad, like the background color. But that's just because the model was only trained on the default args.
## Training using Modal
Assuming you have modal setup (see https://modal.com), it is as simple as running `modal run train_modal.py`, then `python get_modal_data.py` to get the models and tensorboard data from modal. 

But if you wanna close the terminal entirely to do some other stuff, first run `modal run --detach train_modal.py::pretrain_vqvae` which will train the vqvae. Then after some time (check modal to see if it's complete), run `modal run train_modal.py::train_t5`. When both are done, you can get the models and tensorboard from modal as previously stated.
## Training locally
If for some reason you want to do this, first generate a dataset with this command:
```bash
python word_on_background/generator.py \
    --txt-file "word_on_background/CollinsScrabbleWords(2019).txt" \
    --font-path word_on_background/fonts/dejavu-sans-mono.bold.ttf \
    --save-dir dataset \
    --amount 15000
```

Train the vqvae
```bash
cd training/pretraining && python pretrain.py
```

Then train the vqvaet5
```bash
cd .. && python train.py
```
## Random yapping
Like I said in the beginning, initially this was suppose to be trained on a reciept dataset, but I changed because the model at that time sucked and I wanted to pin point if it's the archetecture or the dataset. It was probably the former, or a combination of both cuz the font size of the generated reciepts were pretty small. The model rn combines in order a vqvae, an nn.Embedding, sinusoidal embeddings, two transformer encoder layers, and a t5 decoder. Honestly, I am pretty sure you can drop the vqvae for like a normal convolution or some linear layer and it will run the excatly the same, but I wanted to stick with the vqvae till the very end.

I started from only reading the DONUT model paper and trying to see if I can try implementing it with my own twist, and I am pretty satisfied from what I have  gain while trying to make this. There are two things that vastly sped this up. The first was Claude, I bought my first subsciption mostly because I don't really have anyone irl that I can talk to atm to bounce ideas off of about this, sadly. It's also good for just making templates of logic I already know of but don't want to type. The second that honestly sped this up the most, was training with modal. Training and getting results using my pc's gpu is slow, so this helped out alotttt.

You can see all the notes I made when making this till when I finised in `TIMELINE.md`. Alright, till next time.