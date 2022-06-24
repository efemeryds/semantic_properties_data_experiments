# Semantic properties with corpus data manipulation

## Idea

- Measuring the impact of the evidence in the corpus data on the
encoding of semantic information in the final embedding.


## Reasoning

- Word embeddings may vary depending on the model used, its
parameters and input data.
- There is no one final way of obtaining a numerical representation for a given word.
- Word embeddings are supposed to contain some semantic representation of words. This can be expressed by their location in the vector space. Words close in meaning should be stored
relatively close to each other in the vector space.
- One may try to investigate whether the embeddings encode also more fine-grained information, like
some properties.
- To verify this hypothesis one may prepare a list of concept that often possess given color, a list of concept
that never exhibit given attribut (eg. color) and use very simple model like logistic regression to train it and test on
a given set of word embeddings that represent the list of concepts.
- The assumption here is that if the probing on these embeddings achieves high
score (meaning that is is able to correctly distinguish whether given
concept can exhibit the attribute or not), then we may think (with a grain of salt)
that the semantic information is indeed encoded.


## How to use this repository

First one needs to download the corpus data from http://kyoto.let.vu.nl/cleaned_corpora/ 

The lists of main folders and their purpose is as follows:

- **remove_attribute** - here one can run the *target_color_with_concept* file to
get the corpus data without target color to later use for the embedding training.
Running this script also produces meta-data file that saves the information
on the sentence index and the target word index within the sentence. This information
is crucial to later inject the target back to the chosen ratio of sentences.
One may also run another file *all_colors_with_concept* to remove the group of colors
from the sentences. One should remember that the color is removed only
if it is close (2+- words) from a pre-defined lists of concepts.

- **add_attributes** - here one may run the *get_baselines_embeddings* 
to obtain the *word2vec* model from the data without the target colors or
with all target colors to just prepare some reference for the future results. 
*back_to_the_sentence_target* script allows one to inject back a chosen number
of attributes (eg. black color) and then train the embedding on this
modified data.


- **evaluate_properties** - once all desired embeddings are trained, one
may start the probing and evaluation part. First to get the results
from the logistic regression trained on the previously defined train and test
lists, one needs to run *get_results* file. Then running the *evaluate_results*
produces the final metrics scores (f1, precision and recall) and saves them
to the *evaluation* folder. 

