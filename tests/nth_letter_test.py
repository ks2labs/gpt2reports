import os
import json
import torch
import random
import requests
from tqdm import trange
import numpy as np
from copy import deepcopy
from multiprocessing import Process

from transformers import AutoModelForCausalLM, AutoTokenizer

import transformers
transformers.logging.set_verbosity_error() # suppress tokenizer warnings

# ------ helper functions
def set_seed(seed):
  if seed is not None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------
print("Loading Started")
# load up the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
tok = AutoTokenizer.from_pretrained("gpt2-xl")
# tok.pad_token = tok.eos_token
# tok.pad_token_id = tok.eos_token_id

# define the device and load model there
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
  device = torch.device("cuda:0")

model = model.to(device)
model.eval()
print("Total parameters:", sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values()))
# ---------


def generate(
  sentence,
  n_steps,
  seed = 4,
  do_sample = True,
  temperature = 0.9,
  top_k = 40,
  top_p=0.95,
  num_beams = 1,
  early_stopping=True,
  num_return_sequences = 10,
):
  data = {k:v.to(model.device) for k,v in tok([sentence], return_tensors = "pt").items()}
  set_seed(seed)
  with torch.no_grad():
    len_ = data["input_ids"].size(-1)
    beam_outputs = model.generate(
        **data,
        do_sample=do_sample,
        max_length=len_ + n_steps,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        early_stopping=early_stopping,
        num_return_sequences=num_return_sequences,
        num_beams=num_beams
    )
  generations = [
    tok.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    for beam_output in beam_outputs
  ]
  return generations


def get_sample(question, examples, questions):
  t = f"{question}\n"
  for x in examples:
    t = t + f"{x}\n"
  prompts = []
  for q in questions:
    prompts.append(t + q[:-2])
  return prompts


def run_and_store_result(p, gen_config):
  corr = 0
  for x in generate(p, 1, **gen_config):
    qline = x.split("\n")[len(p.split("\n")) - 1]
    in_, out_ = qline.split("->")[1:]
    if in_.strip()[0] == out_.strip():
      corr += 1
  return corr/10


def one_complete_config(gen_config, ex_idx_list, ex_size = 10):
  results = {}
  print(f"Starting Processing: {ex_size}")
  name = f"nex={ex_size}_" + "_".join([f"{k}={v}" for k,v in gen_config.items()])
  pbar = trange(len(nth_dataset["samples"]))
  for n in pbar:
    pbar.set_description(f"t={n}_{name}")
    this_n_res = []

    idx = ex_idx_list[str((n, ex_size))]  # from predefined list
    train_idx = np.array(idx["train"])
    test_idx = np.array(idx["test"])
    samples = np.array(nth_dataset["samples"][n])

    for qidx in range(3):
      prompts = get_sample(
        question = nth_dataset["questions"][n][qidx],
        examples=samples[train_idx],
        questions=samples[test_idx],
      )

      # run results for this config
      res = []
      for p in prompts:
        res.append(run_and_store_result(p, gen_config))
      this_n_res.append(res)

    results[n] = this_n_res
  
  # determine the name for this results sheet
  
  with open(f"./{name}.json", "w") as f:
    f.write(json.dumps(results))


if __name__ == "__main__":
  set_seed(4)

  with open("./nth_letter.json", "r") as f:
    nth_dataset = json.load(f)

  # step 1: load the experiment parameters
  N_TEST_EXAMPLES = 50
  tempratures = np.linspace(0.5, 1.0, 10)         # this is in linear space
  top_k = (np.linspace(1, 40, 6) * 2).astype(int) # this is in linear space
  num_beams = [1, 4, 8]                           # this is linear space
  n_examples = [1, 4, 8, 12, 16, 20]              # number of examples to give is in linear space

  # just manual experimentation to get top_p
  top_p = np.log(np.linspace(0.95, 1, 10))**5
  top_p -= (min(top_p) - 1e-7)
  top_p /= max(top_p)                              # this is in log space

  # if there is previously stored list of train/test idx load that
  if os.path.exists("./ex_idx_list.json"):
    print("Loading previously stored version of indices")
    with open("./ex_idx_list.json", "r") as f:
      ex_idx_list = json.load(f)

  else:
    # else create a new one
    print("Generating new indices")
    total_n = len(nth_dataset["samples"]) # 10
    ex_idx_list = {}
    for n in range(total_n):
      this_n = nth_dataset["samples"][n]  # 237
      for ex_size in n_examples:
        # take a list, shuffle it and then split first for example and rest for test
        idx = np.arange(len(this_n))
        np.random.shuffle(idx)
        train_idx = idx[:ex_size]
        test_idx = idx[ex_size: ex_size + N_TEST_EXAMPLES]        
        ex_idx_list[str((n, ex_size))] = {"train": train_idx.tolist(), "test": test_idx.tolist()}

    with open("./ex_idx_list.json", "w") as f:
      f.write(json.dumps(ex_idx_list))

  # master loops ---- this program will take days to complete
  for temp in tempratures:
    for tk in top_k:
      for tp in top_p:
        for nb in num_beams:
          gen_config={
            "temperature":float(temp),
            "top_k":int(tk),
            "top_p":float(tp),
            "num_beams":int(nb),
            "early_stopping":True,
            "num_return_sequences":10
          }
          for ex_size in n_examples:
            one_complete_config(gen_config, ex_idx_list, ex_size)
