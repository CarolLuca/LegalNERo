# Named Entity Recognition in Romanian Legal Domain
## Table of contents
* [News](#news)
* [Preview](#preview)
* [Accommodation](#accommodation)
* [Data Sets](#data-sets)
* [Stage 0](#stage-0)
  * [Load the model, or create an empty model](#load-the-model,-or-create-an-empty-model)
  * [Adding Labels or entities](#adding-labels-or-entities)
  * [Training and updating the model](#training-and-updating-the-model)
  * [Calculating prf-values](#calculating-prf-values)
  * [Problems](#problems)
* [Stage 1](#stage-1)
  * [Updates](#updates)
  * [Remarks](#remarks)
  * [Problems](#problems)
* [Stage 2](#stage-2)
* [Behind The Model](#behind-the-model) 
  * [Introduction](#introduction)
  * [Language processing pipelines](#language-processing-pipelines)
  * [Processing text](#processing-text)
  * [How pipelines work](#how-pipelines-work)
  * [Built-in pipeline components](#built-in-pipeline-components)
  * [Word Vectors and Semantic Similarity](#word-vectors-and-semantic-similarity)
  * [Training the named entity recognizer](#training-the-named-entity-recognizer)

## News
* Stage 1 is finished and it can be found in the directory with the same name.
* Stage 2 is planned to be uploaded before 04/28/2021.

## Preview
In this project is developed a named entity recognition model for legal documents in romanian using SpaCy library.
This repository is divided as follows:
* Accommodation - tries before the actual model
* Data Sets - all the data used for the project
* Stage 0 - first functional model with no special training technics
* Stage 1 - under development, will be published before 04/21/2021
* Behind The Model - detailed explanations regarding the model functionality

Each folder contains its own Description.md file in which the content is resumed.

## Accommodation
This part of the project has its own directory in which are presented codes that were used in understanding spaCy usage.

## Data Sets
This folder contains all the data sets used for this project, including two programs for parsing the raw gold data sets.

## Stage 0
This is a brief description of the stage 0 model:
### Load the model, or create an empty model
We can create an empty model and train it with our annotated dataset or we can use existing spacy model and re-train with our annotated data.

```python
if model is not None:
    nlp = spacy.load(model)  # load existing spaCy model
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.blank("en")  # create blank Language class
    print("Created blank 'en' model")

if 'ner' not in nlp.pipe_names :
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)
else :
    ner = nlp.get_pipe("ner")
```
* We can create an empty model using spacy.black(“en”) or we can load the existing spacy model using spacy.load(“model_name”)
* We can check the list of pipeline component names by using nlp.pipe_names() .
* If  we don’t have the entity recogniser in  the pipeline, we will need to create the ner pipeline component using nlp.create_pipe(“ner”) and add that in our model pipeline by using nlp.add_pipe method.
### Adding Labels or entities
```python
# add labels
for _, annotations in train_data:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])

other_pipe = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

# Only training NER
with nlp.disable_pipes(*other_pipe) :
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
```
In order to train the model with our annotated data, we need to add the labels (entities) we want to extract from our text.

* We can add the new entity from our annotated data to the entity recogniser using ner.add_label().
* As we are only focusing on entity extraction, we will disable all other pipeline components to train our model for ner only using nlp.disable_pipes().
### Training and updating the model
```python
for int in range(iteration) :
    print("Starting iteration" + str(int))
    random.shuffle(train_data)
    losses = {}

    for text, annotation in train_data :
        nlp.update(
        [text],
        [annotation],
        drop = 0.2,
        sgd = optimizer,
        losses = losses
        )
  #print(losses)
new_model = nlp
```
* We will train our model for a number of iterations so that the model can learn from it effectively.
* At each iteration, the training data is shuffled to ensure the model doesn’t make any generalisations based on the order of examples.
* We will update the model for each iteration using  nlp.update(). 
### Calculating prf-values
Spacy has a built-in class to evaluate NER. It's called scorer. Scorer uses exact matching to evaluate NER. The precision score is returned as ents_p, the recall as ents_r and the F1 score as ents_f.

The only problem with that is that it returns the score for all the tags together in the document. However, we can call the function only with the TAG we want and get the desired result.
### Problems
* The romanian model "ro_model_news_lg" was not used because of the differences between spaCy v2.2.4. and v3.0.1. Currently, the model is running on the older version, whereas the romanian model offered by spaCy is compatible only with v3.0.1. Why to use this model? It has a couple of overlapping enitity-types and pretrained vectors for them. This problem will be fixed in Stage 2.
* The prf-values are representative for short sentences because there was a problem with the length of phrases from the train data. This problem will be fixed in Stage 1.

## Stage 1
### Updates
* Now it is possible to run the model with decent prf-values for longer texts.
* The only modifications were related to the train data which was recomposed using two auxiliar projects: "Keep Name and Type" (C++) and "GoodIndex" (Java) - both present in the "Data Sets" directory.
### Remarks
* Even though the performance seems to be increased significantly (based on the new prf-values) from Stage 0, it should be kept in mind the fact that the model still has flows.
### Problems
* As aforementioned, beginning with Stage 2, for some entities it will also be used the romanian model "ro_core_news_lg".

## Stage 2
* Will be posted before 04/28/2021.

## Behind The Model
### Introduction
SpaCy’s tagger, parser, text categorizer and many other components are powered by statistical models. Every “decision” these components make – for example, which part-of-speech tag to assign, or whether a word is a named entity – is a prediction based on the model’s current weight values. The weight values are estimated based on examples the model has seen during training. To train a model, you first need training data – examples of text, and the labels you want the model to predict. This could be a part-of-speech tag, a named entity or any other information.

Training is an iterative process in which the model’s predictions are compared against the reference annotations in order to estimate the gradient of the loss. The gradient of the loss is then used to calculate the gradient of the weights through backpropagation. The gradients indicate how the weight values should be changed so that the model’s predictions become more similar to the reference labels over time.

![image](https://user-images.githubusercontent.com/44003293/114925003-3ab0fe80-9e37-11eb-8417-2016202a08d4.png)

* Training data: Examples and their annotations.
* Text: The input text the model should predict a label for.
* Label: The label the model should predict.
* Gradient: The direction and rate of change for a numeric value. Minimising the gradient of the weights should result in predictions that are closer to the reference labels on the training data.

When training a model, we don’t just want it to memorize our examples – we want it to come up with a theory that can be generalized across unseen data. After all, we don’t just want the model to learn that this one instance of “Amazon” right here is a company – we want it to learn that “Amazon”, in contexts like this, is most likely a company. That’s why the training data should always be representative of the data we want to process. A model trained on Wikipedia, where sentences in the first person are extremely rare, will likely perform badly on Twitter. Similarly, a model trained on romantic novels will likely perform badly on legal text.

This also means that in order to know how the model is performing, and whether it’s learning the right things, you don’t only need training data – you’ll also need evaluation data. If you only test the model with the data it was trained on, you’ll have no idea how well it’s generalizing. If you want to train a model from scratch, you usually need at least a few hundred examples for both training and evaluation.

### Language processing pipelines
When you call nlp on a text, spaCy first tokenizes the text to produce a Doc object. The Doc is then processed in several different steps – this is also referred to as the processing pipeline. The pipeline used by the default models consists of a tagger, a parser and an entity recognizer. Each pipeline component returns the processed Doc, which is then passed on to the next component.

![image](https://user-images.githubusercontent.com/44003293/114925043-46042a00-9e37-11eb-9893-4b920010f592.png)

* Name: ID of the pipeline component.
* Component: spaCy’s implementation of the component.
* Creates: Objects, attributes and properties modified and set by the component.

![image](https://user-images.githubusercontent.com/44003293/114922763-bf4e4d80-9e34-11eb-8055-96b49a409282.png)

The processing pipeline always depends on the statistical model and its capabilities. For example, a pipeline can only include an entity recognizer component if the model includes data to make predictions of entity labels. This is why each model will specify the pipeline to use in its meta data, as a simple list containing the component names:

```python
 "pipeline": ["tagger", "parser", "ner"]
```
### Processing text
When you call nlp on a text, spaCy will tokenize it and then call each component on the Doc, in order. It then returns the processed Doc that you can work with.

```python
 doc = nlp("This is a text")
```
When processing large volumes of text, the statistical models are usually more efficient if you let them work on batches of texts. spaCy’s nlp.pipe method takes an iterable of texts and yields processed Doc objects. The batching is done internally.

```python
 texts = ["This is a text", "These are lots of texts", "..."]
 - docs = [nlp(text) for text in texts]
 + docs = list(nlp.pipe(texts))
```
In this example, we’re using nlp.pipe to process a (potentially very large) iterable of texts as a stream. Because we’re only accessing the named entities in doc.ents (set by the ner component), we’ll disable all other statistical components (the tagger and parser) during processing. nlp.pipe yields Doc objects, so we can iterate over them and access the named entity predictions:

```python
import spacy

texts = [
    "Net income was $9.4 million compared to the prior year of $2.7 million.",
    "Revenue exceeded twelve billion dollars, with a loss of $1b.",
]

nlp = spacy.load("en_core_web_sm")
for doc in nlp.pipe(texts, disable=["tagger", "parser"]):
    # Do something with the doc here
    print([(ent.text, ent.label_) for ent in doc.ents])
```
### How pipelines work
spaCy makes it very easy to create your own pipelines consisting of reusable components – this includes spaCy’s default tagger, parser and entity recognizer, but also your own custom processing functions. A pipeline component can be added to an already existing nlp object, specified when initializing a Language class, or defined within a model package.

When you load a model, spaCy first consults the model’s meta.json. The meta typically includes the model details, the ID of a language class, and an optional list of pipeline components. spaCy then does the following:

* Load the language class and data for the given ID via get_lang_class and initialize it. The Language class contains the shared vocabulary, tokenization rules and the language-specific annotation scheme.
* Iterate over the pipeline names and create each component using create_pipe, which looks them up in Language.factories.
* Add each pipeline component to the pipeline in order, using add_pipe.
* Make the model data available to the Language class by calling from_disk with the path to the model data directory

meta.json (excerpt):

```python
{
  "lang": "en",
  "name": "core_web_sm",
  "description": "Example model for spaCy",
  "pipeline": ["tagger", "parser", "ner"]
}
```
So when you call:

```python
 nlp = spacy.load("en_core_web_sm")
```
he model’s meta.json tells spaCy to use the language "en" and the pipeline ["tagger", "parser", "ner"]. spaCy will then initialize spacy.lang.en.English, and create each pipeline component and add it to the processing pipeline. It’ll then load in the model’s data from its data directory and return the modified Language class for you to use as the nlp object.

Fundamentally, a spaCy model consists of three components: the weights, i.e. binary data loaded in from a directory, a pipeline of functions called in order, and language data like the tokenization rules and annotation scheme. All of this is specific to each model, and defined in the model’s meta.json – for example, a Romanian NER model requires different weights, language data and pipeline components than an English parsing and tagging model. This is also why the pipeline state is always held by the Language class. spacy.load puts this all together and returns an instance of Language with a pipeline set and access to the binary data:

"spacy.load under the hood":

```python
lang = "en"
pipeline = ["tagger", "parser", "ner"]
data_path = "path/to/en_core_web_sm/en_core_web_sm-2.0.0"

cls = spacy.util.get_lang_class(lang)   # 1. Get Language instance, e.g. English()
nlp = cls()                             # 2. Initialize it
for name in pipeline:
    component = nlp.create_pipe(name)   # 3. Create the pipeline components
    nlp.add_pipe(component)             # 4. Add the component to the pipeline
nlp.from_disk(model_data_path)          # 5. Load in the binary data
```

When you call nlp on a text, spaCy will tokenize it and then call each component on the Doc, in order. Since the model data is loaded, the components can access it to assign annotations to the Doc object, and subsequently to the Token and Span which are only views of the Doc, and don’t own any data themselves. All components return the modified document, which is then processed by the component next in the pipeline.

"The pipeline under the hood":

```python
doc = nlp.make_doc("This is a sentence")   # create a Doc from raw text
for name, proc in nlp.pipeline:             # iterate over components in order
    doc = proc(doc)                         # apply each component
```
The current processing pipeline is available as nlp.pipeline, which returns a list of (name, component) tuples, or nlp.pipe_names, which only returns a list of human-readable component names.

```python
print(nlp.pipeline)
# [('tagger', <spacy.pipeline.Tagger>), ('parser', <spacy.pipeline.DependencyParser>), ('ner', <spacy.pipeline.EntityRecognizer>)]
print(nlp.pipe_names)
# ['tagger', 'parser', 'ner']
```
### Built-in pipeline components
spaCy ships with several built-in pipeline components that are also available in the Language.factories. This means that you can initialize them by calling nlp.create_pipe with their string names and require them in the pipeline settings in your model’s meta.json.

Usage:

```python
# Option 1: Import and initialize
from spacy.pipeline import EntityRuler
ruler = EntityRuler(nlp)
nlp.add_pipe(ruler)

# Option 2: Using nlp.create_pipe
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)
```
![image](https://user-images.githubusercontent.com/44003293/114986168-57cbe880-9e9c-11eb-83d4-e5c77f54971c.png)

### Disabling and modifying pipeline components
If you don’t need a particular component of the pipeline – for example, the tagger or the parser, you can disable loading it. This can sometimes make a big difference and improve loading speed. Disabled component names can be provided to spacy.load, Language.from_disk or the nlp object itself as a list:

Disable loading:

```python
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser"])
nlp = English().from_disk("/model", disable=["ner"])
```
If you only need a Doc object with named entities, there’s no need to run all pipeline components on it – that can potentially make processing much slower. Instead, you can use the disable keyword argument on nlp.pipe to temporarily disable the components during processing:

Disable for processing:

```python
for doc in nlp.pipe(texts, disable=["tagger", "parser"]):
    # Do something with the doc here
```
If you need to execute more code with components disabled – e.g. to reset the weights or update only some components during training – you can use the nlp.disable_pipes contextmanager. At the end of the with block, the disabled pipeline components will be restored automatically. Alternatively, disable_pipes returns an object that lets you call its restore() method to restore the disabled components when needed.

Disable for block:

```python
# 1. Use as a contextmanager
with nlp.disable_pipes("tagger", "parser"):
    doc = nlp("I won't be tagged and parsed")
doc = nlp("I will be tagged and parsed")

# 2. Restore manually
disabled = nlp.disable_pipes("ner")
doc = nlp("I won't have named entities")
disabled.restore()
```
Finally, you can also use the remove_pipe method to remove pipeline components from an existing pipeline, the rename_pipe method to rename them, or the replace_pipe method to replace them with a custom component entirely.

```python
nlp.remove_pipe("parser")
nlp.rename_pipe("ner", "entityrecognizer")
nlp.replace_pipe("tagger", my_custom_tagger)
```
### Creating custom pipeline components
A component receives a Doc object and can modify it – for example, by using the current weights to make a prediction and set some annotation on the document. By adding a component to the pipeline, you’ll get access to the Doc at any point during processing – instead of only being able to modify it afterwards.

Example:

```python
def my_component(doc):
   # do something to the doc here
   return doc
```

![image](https://user-images.githubusercontent.com/44003293/114986753-0ff99100-9e9d-11eb-87c4-ba9abdc47ea6.png)

Custom components can be added to the pipeline using the add_pipe method. Optionally, you can either specify a component to add it before or after, tell spaCy to add it first or last in the pipeline, or define a custom name. If no name is set and no name attribute is present on your component, the function name is used.

Example:

```python
nlp.add_pipe(my_component)
nlp.add_pipe(my_component, first=True)
nlp.add_pipe(my_component, before="parser")
```
![image](https://user-images.githubusercontent.com/44003293/114986941-46cfa700-9e9d-11eb-8db3-02f4e60703d3.png)

### Example: Pipeline component for entity matching and tagging with custom attributes
This example shows how to create a spaCy extension that takes a terminology list (in this case, single- and multi-word company names), matches the occurrences in a document, labels them as ORG entities, merges the tokens and sets custom is_tech_org and has_tech_org attributes. For efficient matching, the example uses the PhraseMatcher which accepts Doc objects as match patterns and works well for large terminology lists. It also ensures your patterns will always match, even when you customize spaCy’s tokenization rules. When you call nlp on a text, the custom pipeline component is applied to the Doc.

Link: https://github.com/explosion/spaCy/blob/v2.x/examples/pipeline/custom_component_entities.py
```python
#!/usr/bin/env python
# coding: utf8
"""Example of a spaCy v2.0 pipeline component that sets entity annotations
based on list of single or multiple-word company names. Companies are
labelled as ORG and their spans are merged into one token. Additionally,
._.has_tech_org and ._.is_tech_org is set on the Doc/Span and Token
respectively.

* Custom pipeline components: https://spacy.io//usage/processing-pipelines#custom-components

Compatible with: spaCy v2.0.0+
Last tested with: v2.1.0
"""
from __future__ import unicode_literals, print_function

import plac
from spacy.lang.en import English
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span, Token


@plac.annotations(
    text=("Text to process", "positional", None, str),
    companies=("Names of technology companies", "positional", None, str),
)
def main(text="Alphabet Inc. is the company behind Google.", *companies):
    # For simplicity, we start off with only the blank English Language class
    # and no model or pre-defined pipeline loaded.
    nlp = English()
    if not companies:  # set default companies if none are set via args
        companies = ["Alphabet Inc.", "Google", "Netflix", "Apple"]  # etc.
    component = TechCompanyRecognizer(nlp, companies)  # initialise component
    nlp.add_pipe(component, last=True)  # add last to the pipeline

    doc = nlp(text)
    print("Pipeline", nlp.pipe_names)  # pipeline contains component name
    print("Tokens", [t.text for t in doc])  # company names from the list are merged
    print("Doc has_tech_org", doc._.has_tech_org)  # Doc contains tech orgs
    print("Token 0 is_tech_org", doc[0]._.is_tech_org)  # "Alphabet Inc." is a tech org
    print("Token 1 is_tech_org", doc[1]._.is_tech_org)  # "is" is not
    print("Entities", [(e.text, e.label_) for e in doc.ents])  # all orgs are entities


class TechCompanyRecognizer(object):
    """Example of a spaCy v2.0 pipeline component that sets entity annotations
    based on list of single or multiple-word company names. Companies are
    labelled as ORG and their spans are merged into one token. Additionally,
    ._.has_tech_org and ._.is_tech_org is set on the Doc/Span and Token
    respectively."""

    name = "tech_companies"  # component name, will show up in the pipeline

    def __init__(self, nlp, companies=tuple(), label="ORG"):
        """Initialise the pipeline component. The shared nlp instance is used
        to initialise the matcher with the shared vocab, get the label ID and
        generate Doc objects as phrase match patterns.
        """
        self.label = nlp.vocab.strings[label]  # get entity label ID

        # Set up the PhraseMatcher – it can now take Doc objects as patterns,
        # so even if the list of companies is long, it's very efficient
        patterns = [nlp(org) for org in companies]
        self.matcher = PhraseMatcher(nlp.vocab)
        self.matcher.add("TECH_ORGS", None, *patterns)

        # Register attribute on the Token. We'll be overwriting this based on
        # the matches, so we're only setting a default value, not a getter.
        Token.set_extension("is_tech_org", default=False)

        # Register attributes on Doc and Span via a getter that checks if one of
        # the contained tokens is set to is_tech_org == True.
        Doc.set_extension("has_tech_org", getter=self.has_tech_org)
        Span.set_extension("has_tech_org", getter=self.has_tech_org)

    def __call__(self, doc):
        """Apply the pipeline component on a Doc object and modify it if matches
        are found. Return the Doc, so it can be processed by the next component
        in the pipeline, if available.
        """
        matches = self.matcher(doc)
        spans = []  # keep the spans for later so we can merge them afterwards
        for _, start, end in matches:
            # Generate Span representing the entity & set label
            entity = Span(doc, start, end, label=self.label)
            spans.append(entity)
            # Set custom attribute on each token of the entity
            for token in entity:
                token._.set("is_tech_org", True)
            # Overwrite doc.ents and add entity – be careful not to replace!
            doc.ents = list(doc.ents) + [entity]
        for span in spans:
            # Iterate over all spans and merge them into one token. This is done
            # after setting the entities – otherwise, it would cause mismatched
            # indices!
            span.merge()
        return doc  # don't forget to return the Doc!

    def has_tech_org(self, tokens):
        """Getter for Doc and Span attributes. Returns True if one of the tokens
        is a tech org. Since the getter is only called when we access the
        attribute, we can refer to the Token's 'is_tech_org' attribute here,
        which is already set in the processing step."""
        return any([t._.get("is_tech_org") for t in tokens])


if __name__ == "__main__":
    plac.call(main)

    # Expected output:
    # Pipeline ['tech_companies']
    # Tokens ['Alphabet Inc.', 'is', 'the', 'company', 'behind', 'Google', '.']
    # Doc has_tech_org True
    # Token 0 is_tech_org True
    # Token 1 is_tech_org False
    # Entities [('Alphabet Inc.', 'ORG'), ('Google', 'ORG')]
```
Wrapping this functionality in a pipeline component allows you to reuse the module with different settings, and have all pre-processing taken care of when you call nlp on your text and receive a Doc object.

### Word Vectors and Semantic Similarity
Similarity is determined by comparing word vectors or “word embeddings”, multi-dimensional meaning representations of a word. Word vectors can be generated using an algorithm like word2vec and usually look like this:

"banana.vector":

```python
array([2.02280000e-01,  -7.66180009e-02,   3.70319992e-01,
       3.28450017e-02,  -4.19569999e-01,   7.20689967e-02,
      -3.74760002e-01,   5.74599989e-02,  -1.24009997e-02,
       5.29489994e-01,  -5.23800015e-01,  -1.97710007e-01,
      -3.41470003e-01,   5.33169985e-01,  -2.53309999e-02,
       1.73800007e-01,   1.67720005e-01,   8.39839995e-01,
       5.51070012e-02,   1.05470002e-01,   3.78719985e-01,
       2.42750004e-01,   1.47449998e-02,   5.59509993e-01,
       1.25210002e-01,  -6.75960004e-01,   3.58420014e-01,
       # ... and so on ...
       3.66849989e-01,   2.52470002e-03,  -6.40089989e-01,
      -2.97650009e-01,   7.89430022e-01,   3.31680000e-01,
      -1.19659996e+00,  -4.71559986e-02,   5.31750023e-01], dtype=float32)
```
Training word vectors (by spaCy):

```
Dense, real valued vectors representing distributional similarity information are now a cornerstone of practical NLP. The most common way to train these vectors is the Word2vec family of algorithms. If you need to train a word2vec model, we recommend the implementation in the Python library Gensim.
```
Models that come with built-in word vectors make them available as the Token.vector attribute. Doc.vector and Span.vector will default to an average of their token vectors. You can also check if a token has a vector assigned, and get the L2 norm, which can be used to normalize vectors.

```python
import spacy

nlp = spacy.load("en_core_web_md")
tokens = nlp("dog cat banana afskfsd")

for token in tokens:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)
```
The words “dog”, “cat” and “banana” are all pretty common in English, so they’re part of the model’s vocabulary, and come with a vector. The word “afskfsd” on the other hand is a lot less common and out-of-vocabulary – so its vector representation consists of 300 dimensions of 0, which means it’s practically nonexistent. If your application will benefit from a large vocabulary with more vectors, you should consider using one of the larger models or loading in a full vector package, for example, en_vectors_web_lg, which includes over 1 million unique vectors.

```
    Text: The original token text.
    has vector: Does the token have a vector representation?
    Vector norm: The L2 norm of the token’s vector (the square root of the sum of the values squared)
    OOV: Out-of-vocabulary
```
spaCy is able to compare two objects, and make a prediction of how similar they are. Predicting similarity is useful for building recommendation systems or flagging duplicates. For example, you can suggest a user content that’s similar to what they’re currently looking at, or label a support ticket as a duplicate if it’s very similar to an already existing one.

Each Doc, Span and Token comes with a .similarity() method that lets you compare it with another object, and determine the similarity. Of course similarity is always subjective – whether “dog” and “cat” are similar really depends on how you’re looking at it. spaCy’s similarity model usually assumes a pretty general-purpose definition of similarity.

```python
import spacy

nlp = spacy.load("en_core_web_md")  # make sure to use larger model!
tokens = nlp("dog cat banana")

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))
```
In this case, the model’s predictions are pretty on point. A dog is very similar to a cat, whereas a banana is not very similar to either of them. Identical tokens are obviously 100% similar to each other (just not always exactly 1.0, because of vector math and floating point imprecisions).

### Customizing word vectors
Word vectors let you import knowledge from raw text into your model. The knowledge is represented as a table of numbers, with one row per term in your vocabulary. If two terms are used in similar contexts, the algorithm that learns the vectors should assign them rows that are quite similar, while words that are used in different contexts will have quite different values. This lets you use the row-values assigned to the words as a kind of dictionary, to tell you some things about what the words in your text mean.

Word vectors are particularly useful for terms which aren’t well represented in your labelled training data. For instance, if you’re doing named entity recognition, there will always be lots of names that you don’t have examples of. For instance, imagine your training data happens to contain some examples of the term “Microsoft”, but it doesn’t contain any examples of the term “Symantec”. In your raw text sample, there are plenty of examples of both terms, and they’re used in similar contexts. The word vectors make that fact available to the entity recognition model. It still won’t see examples of “Symantec” labelled as a company. However, it’ll see that “Symantec” has a word vector that usually corresponds to company terms, so it can make the inference.

In order to make best use of the word vectors, you want the word vectors table to cover a very large vocabulary. However, most words are rare, so most of the rows in a large word vectors table will be accessed very rarely, or never at all. You can usually cover more than 95% of the tokens in your corpus with just a few thousand rows in the vector table. However, it’s those 5% of rare terms where the word vectors are most useful. The problem is that increasing the size of the vector table produces rapidly diminishing returns in coverage over these rare terms.

### Training the named entity recognizer
All spaCy models support online learning, so you can update a pretrained model with new examples. You’ll usually need to provide many examples to meaningfully improve the system — a few hundred is a good start, although more is better.

You should avoid iterating over the same few examples multiple times, or the model is likely to “forget” how to annotate other examples. If you iterate over the same few examples, you’re effectively changing the loss function. The optimizer will find a way to minimize the loss on your examples, without regard for the consequences on the examples it’s no longer paying attention to. One way to avoid this “catastrophic forgetting” problem (https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting) is to “remind” the model of other examples by augmenting your annotations with sentences annotated with entities automatically recognized by the original model. Ultimately, this is an empirical process: you’ll need to experiment on your data to find a solution that works best for you.

### Updating the Named Entity Recognizer
This example shows how to update spaCy’s entity recognizer with your own examples, starting off with an existing, pretrained model, or from scratch using a blank Language class. To do this, you’ll need example texts and the character offsets and labels of each entity contained in the texts.

Link: https://github.com/explosion/spaCy/blob/v2.x/examples/training/train_ner.py

```python
#!/usr/bin/env python
# coding: utf8
"""Example of training spaCy's named entity recognizer, starting off with an
existing model or a blank model.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.0.0+
Last tested with: v2.2.4
"""
from __future__ import unicode_literals, print_function

import plac
import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


# training data
TRAIN_DATA = [
    ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),
    ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, output_dir=None, n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
            print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == "__main__":
    plac.call(main)

    # Expected output:
    # Entities [('Shaka Khan', 'PERSON')]
    # Tokens [('Who', '', 2), ('is', '', 2), ('Shaka', 'PERSON', 3),
    # ('Khan', 'PERSON', 1), ('?', '', 2)]
    # Entities [('London', 'LOC'), ('Berlin', 'LOC')]
    # Tokens [('I', '', 2), ('like', '', 2), ('London', 'LOC', 3),
    # ('and', '', 2), ('Berlin', 'LOC', 3), ('.', '', 2)]

```
#### Step by step guide
* Load the model you want to start with, or create an empty model using spacy.blank with the ID of your language. If you’re using a blank model, don’t forget to add the entity recognizer to the pipeline. If you’re using an existing model, make sure to disable all other pipeline components during training using nlp.disable_pipes. This way, you’ll only be training the entity recognizer.
* Shuffle and loop over the examples. For each example, update the model by calling nlp.update, which steps through the words of the input. At each word, it makes a prediction. It then consults the annotations to see whether it was right. If it was wrong, it adjusts its weights so that the correct action will score higher next time.
* Save the trained model using nlp.to_disk.
* Test the model to make sure the entities in the training data are recognized correctly.

### Training an additional entity type
This script shows how to add a new entity type ANIMAL to an existing pretrained NER model, or an empty Language class. To keep the example short and simple, only a few sentences are provided as examples. In practice, you’ll need many more — a few hundred would be a good start. You will also likely need to mix in examples of other entity types, which might be obtained by running the entity recognizer over unlabelled sentences, and adding their annotations to the training set.

Link: https://github.com/explosion/spaCy/blob/v2.x/examples/training/train_new_entity_type.py

```python
#!/usr/bin/env python
# coding: utf8
"""Example of training an additional entity type

This script shows how to add a new entity type to an existing pretrained NER
model. To keep the example short and simple, only four sentences are provided
as examples. In practice, you'll need many more — a few hundred would be a
good start. You will also likely need to mix in examples of other entity
types, which might be obtained by running the entity recognizer over unlabelled
sentences, and adding their annotations to the training set.

The actual training is performed by looping over the examples, and calling
`nlp.entity.update()`. The `update()` method steps through the words of the
input. At each word, it makes a prediction. It then consults the annotations
provided on the GoldParse instance, to see whether it was right. If it was
wrong, it adjusts its weights so that the correct action will score higher
next time.

After training your model, you can save it to a directory. We recommend
wrapping models as Python packages, for ease of deployment.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.1.0+
Last tested with: v2.2.4
"""
from __future__ import unicode_literals, print_function

import plac
import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


# new entity label
LABEL = "ANIMAL"

# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting
TRAIN_DATA = [
    (
        "Horses are too tall and they pretend to care about your feelings",
        {"entities": [(0, 6, LABEL)]},
    ),
    ("Do they bite?", {"entities": []}),
    (
        "horses are too tall and they pretend to care about your feelings",
        {"entities": [(0, 6, LABEL)]},
    ),
    ("horses pretend to care about your feelings", {"entities": [(0, 6, LABEL)]}),
    (
        "they pretend to care about your feelings, those horses",
        {"entities": [(48, 54, LABEL)]},
    ),
    ("horses?", {"entities": [(0, 6, LABEL)]}),
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, new_model_name="animal", output_dir=None, n_iter=30):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    ner.add_label(LABEL)  # add new entity label to entity recognizer
    # Adding extraneous labels shouldn't mess anything up
    ner.add_label("VEGETABLE")
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            batches = minibatch(TRAIN_DATA, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
            print("Losses", losses)

    # test the trained model
    test_text = "Do you like horses?"
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == "__main__":
    plac.call(main)
```
#### Step by step guide
* Load the model you want to start with, or create an empty model using spacy.blank with the ID of your language. If you’re using a blank model, don’t forget to add the entity recognizer to the pipeline. If you’re using an existing model, make sure to disable all other pipeline components during training using nlp.disable_pipes. This way, you’ll only be training the entity recognizer.
* Add the new entity label to the entity recognizer using the add_label
method. You can access the entity recognizer in the pipeline via nlp.get_pipe('ner').
* Loop over the examples and call nlp.update, which steps through the words of the input. At each word, it makes a prediction. It then consults the annotations, to see whether it was right. If it was wrong, it adjusts its weights so that the correct action will score higher next time.
* Save the trained model using nlp.to_disk.
* Test the model to make sure the new entity is recognized correctly
