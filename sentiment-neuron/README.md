# Sentiment-neuron

In this test we recreate the [unsupervised sentiment neuron experiment](https://openai.com/blog/unsupervised-sentiment-neuron/) from OpenAI. We wanted to see if our results come out the same or not, at this our `full sequence classifier` works better than, `GPT-finetuned classifier`.

- [GPT-Classification-tests](./GPT-Classification-tests.ipynb): This has the code for testing and a cool Visualisation trick (it doesn't show this on Github, download to run).

<img src="./color-vis.png">

## Results

All Classes are balanced:
```
>>> Counter(np.array(amazon_reviews["train"]["stars"]).astype(int))
Counter({1: 40000, 2: 40000, 3: 40000, 4: 40000, 5: 40000})
```

### M1: Full Sequence Classifier (CLIP-text-encoder)

<img src="./m1.png">


### M2: GPT-Finetuned Classifier

<img src="./m2.png">

## Artify 

Shitposting to a new level!

<img src="./artify.gif">

