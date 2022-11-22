from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import TextDataset,DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline

def load_dataset(train_path,test_path,tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)

    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset,test_dataset,data_collator

# vocab_file="models/exomachina/vocab.json"
# merges_file="models/exomachina/merges.txt"
#tokenizer = GPT2Tokenizer(vocab_file=vocab_file, merges_file=merges_file, model_max_length=1024)
# # change config to match tokenizer
# model.config.vocab_size = len(tokenizer)
# model.resize_token_embeddings(len(tokenizer))
# model.config.pad_token_id = tokenizer.eos_token_id
# model.config.eos_token_id = tokenizer.eos_token_id
# model.config.bos_token_id = tokenizer.bos_token_id



# train from scratch
#model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# train from checkpoint
model = GPT2LMHeadModel.from_pretrained("./gpt2-exomachina/checkpoint-380000", pad_token_id=tokenizer.eos_token_id)

print(model.config)

train_path = 'abstracts.txt'
test_path = 'abstracts_short.txt'
model_dir = 'gpt2-exomachina'
train_dataset,test_dataset,data_collator = load_dataset(train_path,test_path,tokenizer)

# how are the training steps calculated?
# https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments


training_args = TrainingArguments(
    output_dir=model_dir, #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=100, # number of training epochs, 50 = ~13 hours
    per_device_train_batch_size=16, # batch size for training
    per_device_eval_batch_size=4,  # batch size for evaluation
    eval_steps = 10000, # Number of update steps between two evaluations.
    save_steps=10000, # after # steps model is saved
    logging_steps=10000,
    save_total_limit=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# checkpoints are saved every 1000 steps
trainer.train()

"""Test a checkpoint """

 # load the model
model = GPT2LMHeadModel.from_pretrained("./gpt2-exomachina/checkpoint-50000", pad_token_id=tokenizer.eos_token_id)
prompt = "The surface of Mars is"

# generate text
nlp = pipeline('text-generation', model=model, tokenizer=tokenizer)
texts = nlp(prompt, max_length=50, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=3)

for text in texts:
    print(text['generated_text']+"\n")

    # create encoding for the generated text
    encoding = tokenizer.encode(text['generated_text'])

    # create embedding for the generated text
    embedding = model.transformer.wte(torch.tensor([encoding]).to(model.device))

    # get the last embedding (768 for GPT-2)
    last_embedding = embedding[0][-1]

    # get the embedding for the prompt
    prompt_embedding = embedding[0][0]

    # calculate the cosine similarity
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    similarity = cos(last_embedding, prompt_embedding)
    print(similarity)