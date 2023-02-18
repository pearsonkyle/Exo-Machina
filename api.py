from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from typing import List

from transformers import pipeline
from transformers import GPT2TokenizerFast
from transformers import AutoTokenizer, OPTForCausalLM


# load model
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
llm = pipeline('text-generation',model='pearsonkyle/gpt2-arxiv', tokenizer=tokenizer)

#tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-125m")
#llm = pipeline('text-generation',model='facebook/galactica-125m', tokenizer=tokenizer)


# class to store the current model, tokenizer, and pipeline
class Prompter:
    def __init__(self, model_name, tokenizer_name):
        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        self.llm = pipeline('text-generation',model=model_name, tokenizer=self.tokenizer)

    def load(self, model_name, tokenizer_name):
        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        self.llm = pipeline('text-generation',model=model_name, tokenizer=self.tokenizer)

    def prompt(self, prompt, temperature=0.9, max_length=100, do_sample=True, penalty_alpha=0.65, top_k=40, top_p=0.95, num_return_sequences=5, repetition_penalty=1.15):
        texts = self.llm(prompt, temperature=temperature, max_length=max_length, do_sample=do_sample, penalty_alpha=penalty_alpha, top_k=top_k, top_p=top_p, num_return_sequences=num_return_sequences, repetition_penalty=repetition_penalty)
        return texts


app = FastAPI(title="Prompter", description="GPT powered prediction",
    version="0.1.0",
    terms_of_service="",
    contact = {
        "name": "GPT Prompter",
        "url": "https://github.com",
        "email": "kyle.sonofpear@gmail.com",
    },
    #license_info = {
    #    "name": "MIT License",
    #    "url": "https://opensource.org/licenses/MIT",
    #}
)

class PromptInput(BaseModel):
    # hugging face prompt and parameters
    prompt: str = "test"
    temperature: float = 0.85
    max_tokens: int = 40    # max length of context for generation
    extra_tokens: int = 10  # number of extra tokens to generate
    do_sample: bool = True
    penalty_alpha: float = 0.65
    top_k: int = 40
    top_p: float = 0.5
    num_return_sequences: int = 4
    repetition_penalty: float = 1.25
    replace_prompt: bool = True

class PromptList(BaseModel):
    # response to user
    prompts: List[str]
    # TODO: add more info, link back to input?


#@app.get("/")
#def read_root():
#    return RedirectResponse(url="/docs")


@app.post("/predict")
def predict(prompt_input: PromptInput):

    # compute number of tokens in prompt
    tokens = tokenizer.encode(prompt_input.prompt)
    prompt_tokens = len(tokens)

    # trim based on max tokens
    if prompt_tokens > prompt_input.max_tokens:
        trim_tokens = tokens[prompt_tokens-prompt_input.max_tokens:]
        prompt_input.prompt = tokenizer.decode(trim_tokens)
        prompt_tokens = prompt_input.max_tokens

    # generate text
    texts = llm(prompt_input.prompt, temperature=prompt_input.temperature,
                do_sample=prompt_input.do_sample, max_length=prompt_tokens+prompt_input.extra_tokens,
                penalty_alpha=prompt_input.penalty_alpha, top_k=prompt_input.top_k, 
                top_p=prompt_input.top_p, num_return_sequences=prompt_input.num_return_sequences, 
                repetition_penalty=prompt_input.repetition_penalty)

    # format response
    prompt_list = []
    for i in range(prompt_input.num_return_sequences):
        if prompt_input.replace_prompt:
            prompt_list.append(texts[i]['generated_text'].replace(prompt_input.prompt,""))
        else:
            prompt_list.append(texts[i]['generated_text'])
    
    return PromptList(prompts=prompt_list)



# uvicorn api:app --reload