# Stage 3
This stage successfully integrates word embeddings. Let's clarify some aspects about it:
## Differences from Stage 2
* In order to include word embeddings, the base_config.cfg file generated on spaCy's website should prioritise "accuracy".
* So, this time, we construct our base_config.cfg file as follows:

![image](https://user-images.githubusercontent.com/44003293/118612996-fbbff100-b7c6-11eb-9cf1-d2dceb95efe3.png)

* For a better understanding, in this stage were used the word embeddings from spaCy - romanian language (also used in the preparation of ro_core_news_lg).
* This did not require work on the word embeddings and format of the file, the purpose being to see how the model reaction to the introduction of pretrained variables.
* The model is used in the same way as the one from Stage 2.
