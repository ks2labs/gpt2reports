import os
import json
import torch
import random
import numpy as np
import transformers
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

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
model_str = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_str)
tok = AutoTokenizer.from_pretrained(model_str)

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


def run_and_store_result(p, q, gen_config):
  corr = 0
  for x in generate(p, 1, **gen_config):
    pred = x.split("->")[-1].strip()
    target = q.split("->")[-1].strip()
    # print("--------")
    # print("prediction:", pred)
    # print("PP", x)
    # print("QQ:", q)
    # print("target:", target)
    if pred == target:
      corr += 1
  # exit()
  return corr/gen_config["num_return_sequences"]


def one_complete_config(gen_config, ex_idx_list, ex_size = 10):
  # name for this result sheet as well
  name = f"nex={ex_size}_" + "_".join([f"{k}={v}" for k,v in gen_config.items()])
  results = {} # dict with data
  pbar = trange(len(nth_dataset["samples"]))
  for n in pbar:
    pbar.set_description(f"t={n}_{name}")
    this_n_res = []

    idx = ex_idx_list[str((n, ex_size))]  # from predefined list
    train_idx = np.array(idx["train"])
    test_idx = np.array(idx["test"])
    samples = np.array(nth_dataset["samples"][n])

    for qidx in range(N_QUESTIONS):
      question = nth_dataset["questions"][n][qidx]
      prompts = get_sample(
        question=question,
        examples=samples[train_idx],
        questions=samples[test_idx],
      )

      # run results for this config
      res = []
      for i,p in enumerate(prompts):
        res.append(run_and_store_result(p, samples[test_idx][i], gen_config))
      this_n_res.append(res)

    results[n] = this_n_res
  
  with open(f"./{name}.json", "w") as f:
    f.write(json.dumps(results))

if __name__ == "__main__":
  set_seed(4)

  with open("./nth_letter.json", "r") as f:
    nth_dataset = json.load(f)

  # step 1: load the experiment parameters
  N_TEST_EXAMPLES = 50                # with upto 20 examples and 50 test cases
  N_QUESTIONS = 1                     # only check for 1st question as that is clear enough
  num_beams = [1, 5]                  # number of beams for searching
  tempratures = [0.8, 0.9, 0.95, 1.0] # temprature value
  top_k = [2, 10, 25, 40]             # top_k value in generation
  top_p = [0.8, 0.9, 0.95, 0.99]      # top_p value in generation
  n_examples = [1, 5, 10, 15, 20]     # number of examples for priming

  # step 2: load indices
  if os.path.exists("./ex_idx_list.json"):
    # there is previously stored list of train/test idx
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

  # step 3: master loops ---- this program will take a few days to complete
  for nb in num_beams:               # 2x
    for temp in tempratures:         # 4x
      for tk in top_k:               # 4x
        for tp in top_p:             # 4x
          for ex_size in n_examples: # 5x
            # so total multiplier    = 640x
            gen_config={
              "temperature":float(temp),
              "top_k":int(tk),
              "top_p":float(tp),
              "num_beams":int(nb),
              "early_stopping":True,
              "num_return_sequences":10
            }
            one_complete_config(gen_config, ex_idx_list, ex_size)
