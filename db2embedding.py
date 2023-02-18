import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

from database import Database, PAPERentry

device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(device)

# TODO try with GPT2 model


db = Database.load('settings.json', dtype=PAPERentry)
print('querying database...')

entrys = db.session.query(PAPERentry.title,PAPERentry.abstract,PAPERentry.bibcode).all()
#.filter(PAPERentry.vec!="")

batch_size = 128 # ~6GB memory on GPU

# loop over entrys in batches
for i in tqdm(range(0, len(entrys), batch_size)):
    # get batch
    batch = entrys[i:i+batch_size]

    # get text
    texts = [ f"{abstract}" for _, abstract, _ in batch ]

    # tokenize text
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    # get embeddings
    with torch.no_grad():
        outputs = model(**inputs)

        # The output[0] is usually [batch, maxlen, hidden_state], it can be narrowed down to [batch, 1, hidden_state] for [CLS] token, 
        # as the [CLS] token is 1st token in the sequence. Here , [batch, 1, hidden_state] can be equivalently considered as [batch, hidden_state].
        # Since BERT is transformer based contextual model, the idea is [CLS] token would have captured the entire context and 
        # would be sufficient for simple downstream tasks such as classification. Hence, for tasks such as classification using 
        # sentence representations, you can use [batch, hidden_state]. However, you can also consider [batch, maxlen, hidden_state], 
        # average across maxlen dimension to get averaged embeddings. However, some sequential tasks, such as classification 
        # using CNN or RNN requires, sequence of representations, during which you have to rely on [batch, maxlen, hidden_state]. 
        # Also, some training objectives such as predicting the masked words, or for SQUAD 1.1 (as shown in BERT paper), 
        # the entire sequence of embeddings [batch, maxlen, hidden_state] are used.
        embeddings = outputs[1].cpu().detach().numpy() # outputs is 768 dimension

    # update vec in db
    for j in range(len(batch)):
        #title, abstract, vec, bibcode = batch[j]

        # comma delimited string rounding to nearest 6th decimal
        vec = ",".join([str(round(x,6)) for x in embeddings[j]])

        # update entry
        db.session.query(PAPERentry).filter(PAPERentry.bibcode==batch[j][2]).update({PAPERentry.vec:vec})

    # commit changes
    db.session.commit()

db.close()