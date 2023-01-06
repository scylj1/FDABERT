import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import time
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fdabert import parse_args, set_params
from utils import initialise
import dill   

def test(args, model, eval_dataloader, device ):
    loss = 0   
    model.eval()
    losses = []
    print("***** Running evaluating *****")
    print(f"  Num examples = {len(eval_dataloader)}")
    start = time.perf_counter()
    for step, batch in enumerate(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        loss += outputs.loss.item()
        if step % 1000 == 0:
            print("step: {}, eval loss: {}".format(step, loss))
    end = time.perf_counter()
    eval_time = round(end-start)
    try:
        #eval_loss = torch.mean(losses)
        eval_loss = loss / len(eval_dataloader)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    print("perplexity: {}, eval loss: {}, eval time (s): {}".format(perplexity, eval_loss, eval_time))
                  
    return float(eval_loss), perplexity

if __name__ == "__main__":

    # parse input arguments
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = initialise(args)
    
    with open('data_real/eval_dataloader.pkl','rb') as f:
        eval_dataloader = dill.load(f)
  
    model.to(device)
    loss, perplexity = test(args, model, eval_dataloader, device)

    # return statistics
    print ("perplexity{}", perplexity)