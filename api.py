from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from typing import List

from transformers import pipeline
from transformers import GPT2TokenizerFast

from database import Database, PaperEntry

# load model for predictive keyboard
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
llm = pipeline('text-generation',model='pearsonkyle/gpt2-arxiv', tokenizer=tokenizer)

# load database + stuff for nearest neighbor search
db = Database.load('settings.json', dtype=PaperEntry, nearest_neighbor_search=True)

# query all entries and load into memory for fast access
print('querying database...')
entrys = db.session.query(PaperEntry.title,PaperEntry.abstract,PaperEntry.bibcode).all()

# set up fastapi app
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
    temperature: float = 0.75
    max_tokens: int = 40    # max length of context for generation
    extra_tokens: int = 10  # number of extra tokens to generate
    do_sample: bool = True
    penalty_alpha: float = 0.65
    top_k: int = 40
    top_p: float = 0.75
    num_return_sequences: int = 4
    repetition_penalty: float = 1.15
    replace_prompt: bool = True

class PromptList(BaseModel):
    # response to user
    prompts: List[str]
    # TODO: add more info, link back to input?

class SearchInput(BaseModel):
    # search parameters
    query: str
    num_results: int = 10

class Paper(BaseModel):
    # paper info
    title: str
    abstract: str
    bibcode: str

class PaperList(BaseModel):
    # response to user
    papers: List[Paper]

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

@app.post("/search")
def search(search_input: SearchInput):
    # search database for nearest neighbors
    #papers = db.search(search_input.query, search_input.num_results)
    ids = db.SearchFunction(search_input.query, search_input.num_results)
    # entrys is a list of tuples (title, abstract, bibcode)
    papers = [Paper(title=entrys[i][0], abstract=entrys[i][1], bibcode=entrys[i][2]) for i in ids]
    return PaperList(papers=papers)

# uvicorn api:app --reload