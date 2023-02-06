from transformers import GPT2LMHeadModel, GPT2TokenizerFast, pipeline
import torch

"""Test a checkpoint """

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("pearsonkyle/gpt2-arxiv", pad_token_id=tokenizer.eos_token_id)
prompt = "The surface of Mars is"

# generate text
nlp = pipeline('text-generation', model=model, tokenizer=tokenizer)

texts = nlp("Directly imaged exoplanets probe", 
             max_length=50, do_sample=True, num_return_sequences=3, 
             penalty_alpha=0.65, top_k=25, repetition_penalty=1.25,
             temperature=0.9)

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