
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_from_disk
from utils import tokenizedDataset
from model import get_model
from metrics import compute_bleu_score
from tqdm import tqdm

batch_size = 16

#load the previously saved tokenizer to be used for data collator  
tokenizer = AutoTokenizer.from_pretrained("assets/tokenizer")

data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer)

train = load_from_disk('dataset/train')['train']
val = load_from_disk('dataset/val')['train']
test = load_from_disk('dataset/test')['train']

train_data = tokenizedDataset(train)
val_data = tokenizedDataset(val)
test_data = tokenizedDataset(test)

# load model
model = get_model()

def setup():
    dist.init_process_group(backend = 'nccl')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()
    
def train_step():
    #start process
    setup()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda")
    
    model_ = model.to(device)
    ddp_model = DDP(model_, device_ids = [int(os.environ['LOCAL_RANK'])]) # torchrun sets the rank automatically 

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr = 2e-5)
    
    # Distributed Sampler for each dataset to split the data across the GPUs
    train_sampler = DistributedSampler(train_data, rank = rank, num_replicas = world_size) 
    val_sampler = DistributedSampler(val_data, rank = rank, num_replicas = world_size)

    train_dl = DataLoader(train_data, sampler = train_sampler, batch_size = batch_size, collate_fn = data_collator)
    val_dl = DataLoader(val_data, sampler = val_sampler, batch_size = batch_size, collate_fn = data_collator)

    n_epochs = 3

    for epoch in range(n_epochs):
        
        ddp_model.train()
        train_loss = 0.0;
        
        for batch in tqdm(train_dl):
            optimizer.zero_grad()
            outputs = ddp_model(**batch.to(device))
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward() # compute gradients
            optimizer.step() #back propagation

        
        ddp_model.eval()
        val_loss = 0.0
        eval_bleu = torch.tensor(0.0, device = rank)
        
        with torch.no_grad():
            for batch in tqdm(val_dl):
                outputs = ddp_model(**batch.to(device))
                loss = outputs.loss
                labels = batch['labels'].cpu().tolist()
                score = compute_bleu_score(outputs.logits, labels)
                eval_bleu += score['bleu']
                val_loss += loss.item()

        #Reduce the bleu_score across all GPUs to compute the Average score
        dist.all_reduce(eval_bleu, op = dist.ReduceOp.SUM)
        eval_bleu = eval_bleu.item() / world_size
        
        print(f"rank : {rank} Epoch : {epoch+1} \ntrain_loss : {train_loss / len(train_dl)} \nval_loss : {val_loss / len(val_dl)}")
        print(f'eval_bleu {eval_bleu/len(val_dl)}')


    model.save_pretrained("assets/t5-base-sql-gen")
    torch.save(optimizer.state_dict(),"assets/optimizer.pth")
    
    #destroy process
    cleanup()

if __name__ == "__main__":
    train_step()
