# Named Entity Recognition in Romanian Legal Domain
## Table of contents
* [Preview](#preview)
* [Accommodation](#accommodation)
* [Stage 0](#stage-0)
* [Behind the model](#behind-the-model)

## Preview
In this project is developed a named entity recognition model for legal documents in romanian using SpaCy library.
This repository is divided as follows:
* Accommodation - tries before the actual model
* Stage 0 - first functional model with no special training technics
* Behind the model - detailed explanations regarding the model functionality

Every folder contains its own Description.md file in which the content is resumed.

## Accommodation

## Stage 0

## Behind the model
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

