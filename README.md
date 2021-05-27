# Purpose: Matching pre-trained word embedding offset to preprocessed token offset
- This is a project to match the offset of the pre-trained word embeddings such as BERT, ELMo, and Glove to the preprocessed token offset and its dependencies in various ways of preprocessing method such as Stanza and Spacy. This is because BERT uses its own tokenization technique which breaks the word into several subtokens which differ from word-level tokenization. Therefore, by matching the offset of pre-trained results and the preprocessed offset, the result of this can be applied to the graph neural networks without concern of its mismatch of the preprocessed token offset and pre-trained result offset.
- - - -
# Brief description
1. PreTrained2PreProcessed_fronzen.py
 - This is a version where the output is frozen pre-trained word embedding of BERT, ELMo and Glove.
 - Output format
  - emb(tensor): Frozen word embedding of BERT, ELMo and Glove
  - preprocessed_offset_match(list): List of pre-trained word embedding offset of matching with the preprocessed offset
  - preprocessed_dep(list): List of dependency parsing result which is part of the preprocessed result

2. PreTrained2PreProcessed_unfronzen.py
 - This is a version where the output is input of BERT model such as input_ids, attention_mask, token_type_ids, and word embedding of ELMo and Glove.
 - BERT output format
  - input_ids(tensor): Input of BERT model
  - attention_mask(tensor): Input of BERT model
  - token_type_ids(tensor): Input of BERT model
  - preprocessed_offset_match(list): List of pre-trained word embedding offset of matching with the preprocessed offset
  - preprocessed_dep(list): List of dependency parsing result which is part of the preprocessed result
 - ELMo, Glove output format
  - emb(tensor): Frozen word embedding of ELMo and Glove
  - preprocessed_offset_match(list): List of pre-trained word embedding offset of matching with the preprocessed offset
  - preprocessed_dep(list): List of dependency parsing result which is part of the preprocessed result
 
# Prerequisites
## Preprocessing techinques
1. Installing Stanza: https://stanfordnlp.github.io/stanza/#getting-started
2. Installing Spacy: https://spacy.io/usage

## Word embeddings
### BERT
1. Installing BERT: https://huggingface.co/transformers/installation.html

### ELMo
1. Installing ELMo: https://github.com/allenai/allennlp

### Glove
1. Downlaod the pretrained glove word embeddings: https://nlp.stanford.edu/projects/glove/
2. Convert the pretrained glove word embeddings text file format to pickle file format by running "Glove_txt2pkl.py"

# References:
- Stanza: Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020). Stanza: A Python natural language processing toolkit for many human languages. arXiv preprint arXiv:2003.07082.
- Spacy: Matthew Honnibal and Ines Montani. 2017. spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing. To appear (2017).
- BERT: Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
- ELMo: Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., & Zettlemoyer, L. (2018). Deep contextualized word representations. arXiv preprint arXiv:1802.05365.
- Glove: Pennington, J., Socher, R., & Manning, C. D. (2014, October). Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) (pp. 1532-1543).
