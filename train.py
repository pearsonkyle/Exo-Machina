from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import TextDataset,DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch

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

# set device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# train from scratch
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to(device)

print(model.config)

train_path = 'abstracts.txt'
test_path = 'abstracts_short.txt'
model_dir = 'gpt2-arxiv'

train_dataset,test_dataset,data_collator = load_dataset(train_path,test_path,tokenizer)

training_args = TrainingArguments(
    output_dir=model_dir, #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=5, # number of training epochs
    per_device_train_batch_size=8, # batch size for training
    per_device_eval_batch_size=4,  # batch size for evaluation
    eval_steps = 1000, # Number of update steps between two evaluations.
    save_steps=1000, # after # steps model is saved
    logging_steps=1000,
    save_total_limit=10)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# checkpoints are saved every 1000 steps
trainer.train()

# push to cloud
trainer.push_to_hub(model_dir)
tokenizer.push_to_hub(model_dir)