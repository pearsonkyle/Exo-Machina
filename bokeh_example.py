# python -m bokeh serve --show bokeh_example.py
import time
import json
import pickle
from annoy import AnnoyIndex
from transformers import pipeline
from bokeh.models import TextAreaInput, Paragraph, Button, Label, Div
from bokeh.plotting import curdoc
from bokeh.layouts import layout, column, row, gridplot
from transformers import pipeline

from database import Database, ADSEntry
from text_to_vec import spacy_tokenizer

# load database
settings = json.load(open("settings.json", 'r'))
ADSDatabase = Database( settings=settings['database'], dtype=ADSEntry )

# ranks words by importance/occurence
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# reduce dimensionality
pca = pickle.load(open("pca.pkl", "rb"))

# lambda function to process input text
process_input = lambda x: pca.transform(vectorizer.transform([spacy_tokenizer(x)]).toarray())[0]

class TextAI:
    def __init__(self, db, model_checkpoint='./models/checkpoint-195000', neighbor_file=f'test_755.ann'):
        self.db = db
        self.model = pipeline('text-generation',model=model_checkpoint, tokenizer='gpt2', config={'max_length':15})
        ndim = int(neighbor_file.split('_')[-1].split('.')[0])
        self.neighbors = AnnoyIndex(ndim, 'angular')
        self.neighbors.load(neighbor_file) # super fast, will just mmap the file
        self.text = None # to save the last generation
        self.recs = None

    def set_div(self, i):
        self.div.text = self.recs[i][0] + '<br><br>' + self.recs[i][2]
        # add hyperlink to ADS
        self.div.text += f'<br><br><a href="https://ui.adsabs.harvard.edu/abs/{self.recs[i][1]}/abstract" target="_blank">{self.recs[i][1]}</a>'
        self.div_text.text = ' '.join(set(self.recs[i][3].split(' ')))

    def set_text(self, i):
        self.text_input.value += ' '+ self.text[i]

    def __call__(self, x, n=6):
        # generate text 
        text_suggestions = []
        for i in range(n):
            text_suggestions.append(self.model(x[-200:])[0]['generated_text'].replace(x[-200:],'').strip())

        #text = self.model(x, max_length=1000, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)[0]['generated_text']
        # get similar abstracts
        nids = self.neighbors.get_nns_by_vector(process_input(x), 10, search_k=-1, include_distances=False)
        entrys = self.db.query(self.db.dtype.id.in_(nids)) # thread issue with gradio (could cache instead)
        paper_recs = [[entry.title,entry.bibcode,entry.abstract,entry.text] for entry in entrys]
        self.text = text_suggestions
        self.recs = paper_recs
        return text_suggestions, paper_recs

# create text and paper recommendations
textai = TextAI(ADSDatabase)

# create text input 
custom_text = "One of the key drivers of the Mars Exploration Program is the search for evidence of past or present life. In this context, the most relevant martian environments to search for extant life are those associated with geologic units that are depleted in sulfur and"

textai.text_input = TextAreaInput(value=custom_text, width=800, height=500, max_length=1000000)

text_output = Paragraph(text="Some text", width=200, height=100)

# generate some text for the buttons
text, papers = textai(custom_text,6)

# buttons for text auto-complete
auto_buttons = [
    Button(label=text[0], button_type="primary", height=50, width=800),
    Button(label=text[1], button_type="primary", height=50, width=800),
    Button(label=text[2], button_type="primary", height=50, width=800),
    Button(label=text[3], button_type="primary", height=50, width=800),
    Button(label=text[4], button_type="primary", height=50, width=800),
    Button(label=text[5], button_type="primary", height=50, width=800),
]

# add button handler
for i,button in enumerate(auto_buttons):
    button.on_click(lambda x: textai.set_text(i))

# stack of 10 buttons for document recommendations
rec_buttons = [Button(label=f"{papers[i][0]}: {papers[i][1]}", button_type="success", height=50,width=420) for i in range(6)]

def my_text_input_handler(attr, old, new):

    print("my_text_input_handler: attr={0}, old={1}, new={2}".format(attr, old, new))
    text, papers = textai(new,len(auto_buttons))
    for i in range(len(auto_buttons)):
        auto_buttons[i].label = text[i]

    # set recommendations
    for i in range(len(rec_buttons)):
        rec_buttons[i].label = papers[i][0] + '\n' + papers[i][1]

# generate new text and recommendations
textai.text_input.on_change("value", my_text_input_handler)

# div to show abstract recommendations
textai.div = Div(text=""" """, style={"overflow-wrap": "break-word", "width": "800px"}, width=800)
textai.div_text = Div(text=""" """, style={"overflow-wrap": "break-word", "width": "800px"}, width=800)

# create layout
rec_buttons[0].on_click(lambda event: textai.set_div(0))
rec_buttons[1].on_click(lambda event: textai.set_div(1))
rec_buttons[2].on_click(lambda event: textai.set_div(2))
rec_buttons[3].on_click(lambda event: textai.set_div(3))
rec_buttons[4].on_click(lambda event: textai.set_div(4))
rec_buttons[5].on_click(lambda event: textai.set_div(5))

# GUI
curdoc().add_root(layout([
    [textai.text_input, 
        [textai.div, # title+absract
        textai.div_text] # keywords
     ],
     [auto_buttons[0], rec_buttons[0]],
        [auto_buttons[1], rec_buttons[1]],
        [auto_buttons[2], rec_buttons[2]],
        [auto_buttons[3], rec_buttons[3]],
        [auto_buttons[4], rec_buttons[4]],
        [auto_buttons[5], rec_buttons[5]],

]))

curdoc().title = "Exo-Machina"