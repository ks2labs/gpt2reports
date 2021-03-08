<img src="./assets/ks2labsRheader-03.jpg">

# gpt2reports

These are our reports on the generation using GPT2 models from 🤗  to check for robustness in generation.

## Objectives

There are several objectives in this benchmark, as we conduct more research into the task we will add those here.

### Words

Tasks which are related to words:
1. [`nth_letter`](./tests/nth_letter_test.py): The objective is to identify the "n"th letter of each input words and if not available say so
2. `common first letter output`: From the list of words extract those which start with same letter
3. `acversarial common first letter output`: Opposite of `common first letter output`
4. `phrase in word (bool)`: Does a particular phrase exist in the words
5. `phrase in word (extraction)`: Extract the word that has a particular phrase in the words

### Country

1. `extract`: Extract the name of country in each sentence
2. `qa`: Is this country present in this sentence

## Metrics

1. Number of examples to give to complete a task
2. Hardness of a task
3. (for `extraction`) length of extraction
4. parameters in the generation function
5. Structure of prompt: no/short/long prompts
