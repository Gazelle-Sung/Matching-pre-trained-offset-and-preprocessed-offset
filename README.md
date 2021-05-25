# Matching-pre-trained-word-embedding-offset-to-preprocessed-token-offset
This is a project to match the offset of the pre-trained word embeddings such as BERT, ELMo, and Glove to the preprocessed token offset and its dependencies in various ways of preprocessing method such as Stanza and Spacy. Therefore, the result of this can be applied to the graph neural networks without concern of its mismatch of the preprocessed token offset and pre-trained result offset.

# Installation of preprocessing pipelines
1. Stanza: https://stanfordnlp.github.io/stanza/#getting-started
2. Spacy: https://spacy.io/usage

References:
- Stanza: Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020). Stanza: A Python natural language processing toolkit for many human languages. arXiv preprint arXiv:2003.07082.
- Spacy: Matthew Honnibal and Ines Montani. 2017. spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing. To appear (2017).
- BERT: Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
- ELMo: Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., & Zettlemoyer, L. (2018). Deep contextualized word representations. arXiv preprint arXiv:1802.05365.
- Glove: Pennington, J., Socher, R., & Manning, C. D. (2014, October). Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) (pp. 1532-1543).
