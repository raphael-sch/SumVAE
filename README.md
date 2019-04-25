Paper with more information:
https://arxiv.org/abs/1809.05233

# run training
    python train_vae.py TRAIN_FILE data/train/train_example.txt -DEV_DIR data/test/example -VOCAB_FILE data/train/vocab.txt -OUTPUT_DIR outputs/example1 -CONFIG configs/example.yaml

# evaluates model
    python summarize.py outputs/example1/ data/test/example -output_len 18 -batch_size 16

# sample new sentence
    python sampling.py outputs/example1/ sample -output_len 18
