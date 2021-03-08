# What is the data?

In this experiment we have to run a tonne of permutations of different parameters and we need to understand what parameters influence what. This dummy dataset is generated as follows:
```python
N_SAMPLE = 50
results = []
task_names = ["hippo-banana", "night-owl", "pink-toe", "carol-shelby"]
for n in range(4,8,1):
  samples = nth_dataset["samples"][n]           # nth_dataset is the json
  for ex_size in range(1, 15, 4):
    idx = [i for i in range(len(samples))]      # define an array of indices [0, ... , n]
    np.random.shuffle(idx)                      # shuffle so it looks like [23, 34, ... 87]
    train_idx = idx[:ex_size]                   # train indices are first `ex_size` elements
    test_idx = idx[ex_size: ex_size + N_SAMPLE] # test indices are 50 elements after it
    for temp in [0.8, 0.9]:
      for topk in [10, 30]:
        for topp in [0.8, 0.99]:
          for nbeams in [1, 5, 10]:

            # this is how we would get the data
            # for each `test_idx` it calculates the accuracy of the model
            # thus accuracy.shape == text_idx.shape
            accuracy = model(samples, train_idx, test_idx, temp, topk, topp, nbeams)

            # train_string -----> Ignore
            train_string = f"What is the {n}th letter in each word?\n" + \
              "\n".join(samples[train_idx])
            test_strings = samples[test_idx].tolist()

            # for now I am just putting in random values in [0.0, 1.0)
            # accuracy is correct/total so it will anyways be in [0.0, 1.0]
            results.append({
              "accuracy": np.random.random(len(test_idx)),
              "train_string": train_string,  # train idx for viz.
              "test_strings": test_strings,  # test idx for viz.
              "task_id": task_names[n-4],    # this is the name of the task
              "number_of_examples": ex_size, # parameter
              "temprature": temp,            # parameter
              "top_k": topk,                 # parameter
              "top_p": topp,                 # parameter
              "number_of_beams": nbeams,     # parameter
            })
```

|idx|accuracy|train_string|test_strings|task_id|number_of_examples|temprature|top_k|top_p|number_of_beams|
|-|-|-|-|-|-|-|-|-|-|
|12|[0.94,0.41,...]|"What is the 6th letter in each word?\nin -> ti... "|[in -> through; out -> h, in -> interest;|pink-toe|9|0.9|10|0.99|1|
|123|[0.53,0.96,...]|"What is the 4th letter in each word?\nin -> ti... "|[in -> through; out -> h, in -> interest;|pink-toe|9|0.9|10|0.8|10|

