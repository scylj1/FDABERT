import dill
import torch
from utils import fl_partition, initialise, train, test, logger
from fdabert import parse_args

args = parse_args()
# determine device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(1)
model = initialise(args)
model.to(device)
print(2)
with open('data_real/eval_dataloader.pkl','rb') as f:
    eval_dataloader = dill.load(f)
print(3)
#set_params(model, parameters)
loss, perplexity = test(args, model, eval_dataloader, device)
print(loss)
print(perplexity)