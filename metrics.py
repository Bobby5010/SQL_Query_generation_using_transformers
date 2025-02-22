
import torch
import evaluate
from transformers import AutoTokenizer
from typing import List, Dict

tokenizer = AutoTokenizer.from_pretrained("assets/tokenizer") 
bleu_score = evaluate.load("bleu")

def compute_bleu_score(logits, labels : List[List[int]]) -> Dict:
    """
        Accepts raw logits in form of torch.Tensor and computes the BLEU Score.

        The labels by default are  padded using `-100`. 
        We replace this by `tokenizer.pad_token_id` to avoid OverflowError while decoding the labels to compute `bleu_score`  
    """
    
    predictions = torch.argmax(logits, dim =-1).cpu().tolist()
    labels = [[token if token != -100 else tokenizer.pad_token_id for token in label] for label in labels]

    predictions = tokenizer.batch_decode(predictions, skip_special_tokens = True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens = True)

    score = bleu_score.compute(predictions =  predictions, references = [[label] for label in labels])
    return {"bleu" : score['bleu']}
