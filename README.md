# Semantic properties with corpus data manipulation

## Reasoning

- NLP models may be used for various tasks including language translation, search prediction, or providing automatic recommendations. For all of these approches one needs to use word embeddings to encode words into a numerical form.
- Word embeddings may vary depending on the model used, its
parameters and input data. There is no one final way of obtaining a numerical representation for a given word. Word embeddings are supposed to contain some semantic representation of words. This can be expressed by their location in the vector space. Words close in meaning should be stored
relatively close to each other in the vector space. One may try to investigate whether the embeddings encode also more fine-grained information, like
some properties.
- One could for example try to test whether word embeddings contain information in what colors certain objects may appear, eg. that the bear may be brown but not necessarily green. To perform such experiment one may prepare a list of concept that may appear in a given color, a list of
concept that never exhibit given color and use very simple model like logistic regression (as probing) to train it and test on a given set of word embeddings that represent the list of concepts.
- The binary result may be a first step towards claiming how well a given set of word embeddings encodes the concepts and their properties.

## Idea

- One interesting research question could be to verify how many sentences expressing a given concept
and its property (eg. color) need to be passed to the word embedding model so that the information
is later encoded and can be extracted with the use of probing (eg. logistic regression).


## Methodology

### Removing attributes


### Adding attributes


### Evaluation
