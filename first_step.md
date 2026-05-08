# Project Step 0

## Plan

The first step of this project is to attempt to extract a sentiment direction from text. This will be broken down into a few sub-pieces.

### Gathering data for analysis

1. Use the SST-2 dataset https://huggingface.co/datasets/stanfordnlp/sst2 (and withold some for later validation)
2. Use Qwen2.5-0.5B-Instruct
3. Run through a forward pass of the text (without chat templates)
4. For the last non-padding token, turn on `output_hidden_states=True` in `transformers` model
5. Gather the hidden states at those positions and store to disk alongside metadata for that prompt / dataset item (ie positive or negative)

### Performing analysis

1. Load the data from above
2. Perform a simple analysis:
```
direction[layer] = mean_positive[layer] - mean_negative[layer]
direction[layer] = direction[layer] / direction[layer].norm()
```
3. Store the directions per-layer that we think are positive

### Validating analysis

1. Load the data from the analysis (per-layer direction for sentiment)
2. Load a witheld test set from the dataset
3. Do a forward pass of the test data as before and extract the hidden layers for the last non-padding token
4. Validate that our directions do indicate sentiment by scoring `dot(hidden[layer], direction[layer])` and checking for positive values there
5. Store the outcome of the analysis per-layer per-text as well as summarized into a json

## Conclusion

So, the basic idea here is to begin very naive and simple inspection of intermediate state in transformers and validating that simple methods for inspection are performing reasonably well. Once we have successfully accomplished this, we can move on to more advanced analysies, methods, and models. But first, we should ensure that our fundamentals are correct.
