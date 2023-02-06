import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

from database import Database, PAPERentry

device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(device)

db = Database.load('settings.json', dtype=PAPERentry)
print('querying database...')

entrys = db.session.query(PAPERentry.title,PAPERentry.abstract,PAPERentry.vec,PAPERentry.bibcode).all()
#.filter(PAPERentry.vec!="")


batch_size = 128 # ~6GB memory

# loop over entrys in batches of 32
for i in tqdm(range(0, len(entrys), batch_size)):
    # get batch
    batch = entrys[i:i+batch_size]

    # get text
    texts = [ f"{abstract}" for _, abstract, _, _ in batch ]

    # tokenize text
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    # get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs[0][:,0,:].cpu().detach().numpy()
        # outputs is 768 dimension

    # update vec in db
    for j in range(len(batch)):
        #title, abstract, vec, bibcode = batch[j]

        # comma delimited string rounding to nearest 6th decimal
        vec = ",".join([str(round(x,6)) for x in embeddings[j]])

        # update entry
        db.session.query(PAPERentry).filter(PAPERentry.bibcode==batch[j][3]).update({PAPERentry.vec:vec})

    # commit changes
    db.session.commit()

db.close()

