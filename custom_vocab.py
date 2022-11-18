import os
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from transformers import GPT2TokenizerFast

# Create a tokenizer
tokenizer = Tokenizer(models.BPE())

# GPT-2 does not use a normalizer
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# the only special token is the end of text token
trainer = trainers.BpeTrainer(min_frequency=3, special_tokens=["<|endoftext|>"], vocab_size=65536)

# train the tokenizer
tokenizer.train(["abstracts.txt"], trainer=trainer)

# print number of words in vocab
vocab = tokenizer.get_vocab()
print(f"Vocab size: {len(vocab)}")

# save vocab
#tokenizer.save("exomachina_vocab.json")

if not os.path.exists("models"):
    os.mkdir("models")

# save model
if not os.path.exists("models/exomachina"):
    os.makedirs("models/exomachina")

tokenizer.model.save("models/exomachina")

# apply the byte-level post-processing for the GPT-2 tokenizer
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

# byte-level decoder
tokenizer.decoder = decoders.ByteLevel()

# save the tokenizer as GPT2TokenizerFast
wrapped_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer, vocab_file="models/exomachina/vocab.json", merges_file="models/exomachina/merges.txt", model_max_length=1024)

# test the tokenizer
print(wrapped_tokenizer("This is a test"))

# Resources:
# https://huggingface.co/course/chapter6/8?fw=pt#building-a-bpe-tokenizer-from-scratch
# https://huggingface.co/docs/transformers/tokenizer_summary#bytepair-encoding-bpe