# Stage 2
Stage 2 is out and clarifies different aspects.
## Updates
* The transition to spaCy v3 was successful.
* The model can be used from the command line.
## Training
Let's see how can we build a custom NER model in Spacy v3.0, using Spacy’s recommended Command Line Interface (CLI) method instead of the custom training loops that were typical in Spacy v2.
### Overview
Essentially, in Spacy v3, there has been a shift toward training your model pipelines using the spacy train command on the command line instead of making your own training loop in Python. As a result of this, the old data formats (json etc.) that were used in Spacy v2 are no longer accepted and you have to convert your data into a new .spacy format.
### The new .spacy format
In version 2, the format for NER was as follows:

```python
[('The F15 aircraft uses a lot of fuel', {'entities': [(4, 7, 'aircraft')]}),
 ('did you see the F16 landing?', {'entities': [(16, 19, 'aircraft')]}),
 ('how many missiles can a F35 carry', {'entities': [(24, 27, 'aircraft')]}),
 ('is the F15 outdated', {'entities': [(7, 10, 'aircraft')]}),
 ('does the US still train pilots to dog fight?',
  {'entities': [(0, 0, 'aircraft')]}),
 ('how long does it take to train a F16 pilot',
  {'entities': [(33, 36, 'aircraft')]}),
 ('how much does a F35 cost', {'entities': [(16, 19, 'aircraft')]}),
 ('would it be possible to steal a F15', {'entities': [(32, 35, 'aircraft')]}),
 ('who manufactures the F16', {'entities': [(21, 24, 'aircraft')]}),
 ('how many countries have bought the F35',
  {'entities': [(35, 38, 'aircraft')]}),
 ('is the F35 a waste of money', {'entities': [(7, 10, 'aircraft')]})]
```
Spacy v3.0, however, no longer takes this format and this has to be converted to their .spacy format by converting these first in doc and then a docbin.
## How to use it
Stage 2/Stage 2 - v3/example.txt ilustrates the commands necessary for accessing the Stage 2 model.
## Desired modifications
* Implementation of multi-word entities - for output
* Using pretrained word embeddings - from Stage 3
* Modyfing other elements from configuration file
* Working on the TRAIN_DATA and TEST_DATA arrays
## Decision regarding future stages
* As mentioned before, it will not be used the model ro_core_news_lg
* There will be no prf-values for Stage 2, as it is an intermediate state between Stages 1 and 3.
