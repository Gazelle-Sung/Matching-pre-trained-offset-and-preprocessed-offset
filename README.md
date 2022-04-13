# Overview
- This project is to match the offset of the pre-trained word embeddings such as BERT, ELMo, and Glove to the preprocessed token offset and its dependencies in various ways of preprocessing methods such as Stanza and Spacy. This is because BERT uses its own tokenization technique which breaks the word into several subtokens which differ from word-level tokenization. Therefore, by matching the offset of pre-trained results and the preprocessed offset, the result of this can be applied to the graph neural networks without concern of its mismatch of the preprocessed token offset and pre-trained result offset. Moreover, it is not only focused on BERT but other pre-trained word embeddings such as ELMo and Glove for flexibility of selecting embeddings.


# Brief description
- OffsetMatching2Preprocessed.py
> Output format
> - emb (tensor): Token representation of given text.
> - preprocessed_offset_match (list): Pre-trained word embedding offsets of matching with the preprocessed offsets.
> - preprocessed_dep (list): Dependency parsing result which is part of the preprocessed result.

- text_processing.py
> Output format
> - cur_clause_feature (list): Sturcture of token id, token offsets, token, dependency, pos, ner.
 
 
# Prerequisites
> Packages:
> - pickle
> - torch
> - os
> - numpy
> - argparse
> - transformers
> - allennlp

> Preprocessing techinques
> - Installing Stanza: https://stanfordnlp.github.io/stanza/#getting-started
> - Installing Spacy: https://spacy.io/usage

> Word embeddings
>> BERT
>> - Installing BERT: https://huggingface.co/transformers/installation.html

>> ELMo
>> - Installing ELMo: https://github.com/allenai/allennlp

>> Glove
>> - Downlaod the pretrained glove word embeddings: https://nlp.stanford.edu/projects/glove/
>> - Convert the pretrained glove word embeddings text file format to pickle file format by running "Glove_txt2pkl.py"

# Parameters:
> - pipeline(str, defaults to "spacy"): Preprocessing methods such as Stanza or Spacy.
> - emb(str, defaults to "bert"): Pre-trained word embedding such as BERT, ELMo or Glove.
> - output_type(str, defaults to "frozen"): This only affects the result of BERT embeddings so it gives the input format of the BERT model with input_ids, attention_mask, token_type_ids when the output_type is set to "unfrozen" while it gives frozen word embeddings when the output_type is set to default which is "frozen".
> 
# References:
- Stanza: Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020). Stanza: A Python natural language processing toolkit for many human languages. arXiv preprint arXiv:2003.07082.
- Spacy: Matthew Honnibal and Ines Montani. 2017. spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing. To appear (2017).
- BERT: Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
- ELMo: Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., & Zettlemoyer, L. (2018). Deep contextualized word representations. arXiv preprint arXiv:1802.05365.
- Glove: Pennington, J., Socher, R., & Manning, C. D. (2014, October). Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) (pp. 1532-1543).
