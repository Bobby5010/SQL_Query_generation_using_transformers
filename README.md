
<h1>SQL-Query-Generation-using-Transformers</h1>
<p>This project implements SQL query generation using a fine-tuned T5 (Text-to-Text Transfer Transformer) model.
Given a natural language question, the model generates an equivalent SQL query.</p>

This model has been fine tuned on the [wikisiql](https://www.kaggle.com/datasets/shahrukhkhan/wikisql) dataset available on kaggle

<h2>Hardware</h2>

<ul>
  <li>2 T4 GPUs (available on kaggle/colab) for Distributed training</li>
</ul>

<h2>Setup</h2>

<ul>
  <li>Clone the repository and change to the workspace</li>
</ul>

```bash
git clone https://github.com/Bobby5010/SQL_Query_generation_using_transformers.git
cd SQL_Query_generation_using_transformers
```

<h2>Dependencies</h2>

<p>Install the necessary requirements</p>

```bash
pip install -r requirements.txt
```

<h2>Training</h2>

<p>This project uses T5-base from hugging face  as the base model for fine tuning .
During training, we monitored the <b>BLEU score</b> to evaluate the accuracy of the generated SQL queries. 
The model achieved a <b>BLEU score of 0.76</b> whichi indicates high similarity between generated and ground-truth SQL queries.  
</p>

<p>This Project Implements </p>
    
<ul>
    <li>Custom training loop with Pytorch</li>
    <li>multi-GPU training with torch DDP</li>
</ul>

Run model.py
    
<ul>
  <li>This performs tokenization and saves the tokenizer to assets/tokenizer to be used later</li>
</ul>

```bash 
python model.py
```

<p>Run the following in cmd to initiate Training</p>

```bash
torchrun --nnodes=1 --nproc_per_node=2 train.py
```

<p>The training takes around <b>35 minutes</b> for <b>3</b> epochs</p>
<p>The final model and tokenizer states are available in <b>assets</b> folder</p>

<h2>Inference</h2>

The model and optim states can be loaded in the normal manner for further training or for Inference 
These state files haven't been included due to the large size

<p>Loading the model and optimizer</p>

```python

import torch
from transformers import AutoModelForSeq2SeqLM

#load  model state
model = AutoModelForSeq2SeqLM.from_pretrained('assets/t5-base-sql-gen')

optimizer = torch.optim.AdamW(model.parameters(), lr = 0.01)

#load optimizer state
optimizer.load_state_dict(torch.load('assets/optimizer.pth'))

```
