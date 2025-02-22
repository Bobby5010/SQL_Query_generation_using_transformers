
from datasets import load_dataset, DatasetDict
from transformers  import AutoTokenizer, AutoModelForSeq2SeqLM
from config import Config

def get_model():
    return AutoModelForSeq2SeqLM.from_pretrained(Config.model_ckpt)


def tokenize_data(sample):
    return {
        "input_ids" : tokenizer(
            sample['question'],
            return_tensors = 'pt',
            return_attention_mask = False
        )["input_ids"],

        "labels" : tokenizer(
            sample['sql'],
            return_tensors = 'pt',
            return_attention_mask = False
        )["input_ids"]
    }


#perform tokenization and save the dataset and the tokenizer
if __name__ == "__main__":
    train = load_dataset('csv', data_files = Config.train_path)
    val = load_dataset('csv', data_files = Config.val_path)
    test = load_dataset('csv', data_files = Config.test_path)

    tokenizer = AutoTokenizer.from_pretrained(Config.model_ckpt)
    
    train_ = train.map(tokenize_data, num_proc = 4)
    val_ = val.map(tokenize_data, num_proc = 4)
    test_ = test.map(tokenize_data, num_proc = 4)

    tokenizer.save_pretrained("assets/tokenizer") #save the tokenizer to load the same to compute bleu_score
    
    train_.save_to_disk('dataset/train')
    val_.save_to_disk('dataset/val')
    test_.save_to_disk('dataset/test')
