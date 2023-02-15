import argparse
import flwr as fl
from flwr.common.typing import Scalar
import ray
import torch
import torchvision
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple, List
from utils import fl_partition, initialise, train, test, logger #, load_data
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
import dill
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
from freeze import freeze_unfreeze_layers, get_para_num, get_trainable_para_num
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
#RAY_ADDRESS="128.232.115.65:6379"
def parse_args():
    parser = argparse.ArgumentParser(description="Flower Simulation with bert")

    parser.add_argument("--num_client_cpus", type=int, default=3)
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--num_clients", type=int, default=2)
    
    parser.add_argument("--do_freeze", type=bool, default=False)
    
    parser.add_argument(
        "--fed_dir_data",
        type=str,
        default=None,
        help="federated dataset dir.",
    )
     
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--cache_dir", type=str,
        default=None,
        help="Where do you want to store the pretrained models downloaded from huggingface.co"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
        ),
    )
    parser.add_argument(
        "--line_by_line",
        type=bool,
        default=False,
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`train_file` should be a csv, json or txt file.")
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be a csv, json or txt file.")

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args

            
# Flower client, adapted from Pytorch quickstart example
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, fed_dir_data: str, args, freeze_layers):
        self.cid = cid
        self.args = args
        self.args.output_dir = self.args.output_dir + str(int(cid)+1)
        self.fed_dir_data = fed_dir_data
        self.model = initialise(self.args)
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        
        print("Train Data loading...")            
        with open(self.fed_dir_data + 'client{}/train{}_dataloader.pkl'.format(self.args.num_clients, self.cid),'rb') as f:
            self.train_dataloader = dill.load(f)
        print("Train Data loading finished...")
        
        print("Eval Data loading...")
        with open(self.fed_dir_data + 'client{}/eval{}_dataloader.pkl'.format(self.args.num_clients, self.cid),'rb') as f:
            self.eval_dataloader = dill.load(f)
        print("Eval Data loading finished...")
        
        if self.args.do_freeze:
            freeze(self.model, freeze_layers, self.cid)
            
        #print("cuda:{}".format(str(int(cid)+7)))

    def get_parameters(self, config):
        return get_params(self.model)

    def fit(self, parameters, config):
        set_params(self.model, parameters)
       # num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        
        print("Training Started...")
        train(self.args, self.model.to(self.device), self.train_dataloader, self.device)
        print("Training Finished...")
        # Return local model and statistics
        return get_params(self.model), len(self.train_dataloader), {}

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        #num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])     
    
        # Evaluate       
        loss, perplexity = test(self.args, self.model.to(self.device), self.eval_dataloader, self.device)
        print("Evaluating finished...")
        # Return statistics
        return float(loss), len(self.eval_dataloader), {"perplexity": float(perplexity)}
    
def freeze(model, freeze_layers, cid):

    get_para_num(model)
    get_trainable_para_num(model)
    layer_all = 6
    if int(cid) == 0:
        if freeze_layers[0] == 1:
            freeze_unfreeze_layers(model, 0, unfreeze=False)
        else:       
            freeze_unfreeze_layers(model, (0, freeze_layers[0]-1), unfreeze=False)
    else:
        before = sum(freeze_layers[0:int(cid)])
        start = before % layer_all
        if freeze_layers[int(cid)] == 1:
            freeze_unfreeze_layers(model, start, unfreeze=False)
        else: 
            end = start + freeze_layers[int(cid)] - 1
            if end < 6:     
                freeze_unfreeze_layers(model, (start, end), unfreeze=False)
            else: 
                freeze_unfreeze_layers(model, (start, 5), unfreeze=False)
                if end ==6 :
                    freeze_unfreeze_layers(model, 0, unfreeze=False)
                else:
                    freeze_unfreeze_layers(model, (0, end - 6), unfreeze=False)
                
    get_para_num(model)
    get_trainable_para_num(model)
    
def allocate_freeze(train_list, num_clients):
    train_all = sum(train_list)
    layer_all = 12
    freeze_layers = []
    for cid in range(num_clients):
        freeze_layer = round(layer_all * train_list[cid] / train_all)
        if freeze_layer == 0:
            freeze_layer += 1
        if freeze_layer > 4:
            freeze_layer = 4
        freeze_layers.append(freeze_layer)
        
    return freeze_layers


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 1,  # number of local epochs
        "batch_size": 8,
    }
    return config


def get_params(model: torch.nn.ModuleList) -> List[np.ndarray]:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model: torch.nn.ModuleList, params: List[np.ndarray]):
    """Set model weights from a list of NumPy ndarrays."""
    print("###########setting#################")
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    print("###########finish################")


def get_evaluate_fn(
    testset, args
) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        # determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = initialise(args)
        
   
        with open('data_real/eval_dataloader.pkl','rb') as f:
            eval_dataloader = dill.load(f)

        set_params(model, parameters)
        model.to(device)
        loss, perplexity = test(args, model, eval_dataloader, device)

        model.save_pretrained(args.output_dir)
        # return statistics
        return loss, {"perplexity": perplexity}

    return evaluate


# Start simulation (a _default server_ will be created)

if __name__ == "__main__":

    # parse input arguments
    args = parse_args()
    
    pool_size = args.num_clients  # number of dataset partions (= number of total clients)
    client_resources = {
        "num_cpus": 7, # args.num_client_cpus,
        "num_gpus": 1
    }  # each client will get allocated 3 CPUs

    freeze_layers = []
    if args.do_freeze:
        train_list = []
        for i in range (args.num_clients):
            with open(args.fed_dir_data + 'client{}/train{}_dataloader.pkl'.format(args.num_clients, i),'rb') as f:
                train_f = dill.load(f)
                train_list.append(len(train_f))
        print(train_list)
        freeze_layers = allocate_freeze(train_list, args.num_clients)
        print(freeze_layers)
            
    # configure the strategy
    from fedavg import FedAvg
    strategy = FedAvg(
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        min_fit_clients=pool_size,
        min_evaluate_clients=pool_size,
        min_available_clients=pool_size,  # All clients should be available
        on_fit_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn("data/partition/val1.txt", args),  # centralised evaluation of global model
    )

    def client_fn(cid: str):
        # create a single client instance
        return FlowerClient(cid, args.fed_dir_data, args, freeze_layers)

    # (optional) specify Ray config
    ray_init_args = {"include_dashboard": False}

    from app import start_simulation
    # start simulation
    start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args,
    )