import pickle
import torch
import os
import numpy as np

from text_processing import preprocessing


# *** Select a pre-trained Word Embeddings
def sel_pretrained(_lm="bert", _pm="bert-base-uncased"):
    # 1. BERT
    if _lm.lower() == "bert":
        print("****** Selected pretrained word embeddings: {}".format(_lm))
        from transformers import BertModel, BertTokenizerFast
        PRETRAINED_MODEL = _pm
        return BertModel.from_pretrained(PRETRAINED_MODEL), BertTokenizerFast.from_pretrained(PRETRAINED_MODEL)
    
    # 2. RoBERTa
    elif _lm.lower() == "roberta":
        print("****** Selected pretrained word embeddings: {}".format(_lm))
        from transformers import RoBertaModel, RobertaTokenizerFast
        PRETRAINED_MODEL = _pm
        return RoBertaModel.from_pretrained(PRETRAINED_MODEL), RobertaTokenizerFast.from_pretrained(PRETRAINED_MODEL)
    

    # 3. Elmo
    elif _lm.lower() == "elmo":
        print("****** Selected pretrained word embeddings: {}".format(_lm))
        from allennlp.commands.elmo import ElmoEmbedder
        return None, ElmoEmbedder(cuda_device=0)

    # 4. Glove
    elif _lm.lower() == "glove":
        print("****** Selected pretrained word embeddings: {}".format(_lm))
        
        if _pm.lower() == "42b":
            glove_PATH = "./glove/glove.42B.300d.pkl"
        elif _lm.lower() == "glove840b":
            glove_PATH = "./glove/glove.840B.300d.pkl"
            
        with open(glove_PATH, "rb") as file:
            glove = pickle.load(file)
        return None, glove


def embedding2preprocessed(args, datasample, embedder, model):
    # preprocessed result
    preprocessed_result = preprocessing(datasample, args.pipeline)

    if args.emb == "bert":
        # bert tokenized result
        tokenized_result = embedder(datasample, return_offsets_mapping=True, return_tensors='pt')

        for p_idx, p_result in enumerate(preprocessed_result):
            p_offset = p_result.offset
            p_start_offset = p_offset[0]
            p_end_offset = p_offset[1]

            cur_offset_list = []
            for b_idx, b_offset in enumerate(tokenized_result['offset_mapping'][1:-1]):
                b_start_offset = b_offset[0]
                b_end_offset = b_offset[1]

                # offset of the bert tokenized token is in the preprocessed offset
                if b_start_offset >= p_start_offset and b_end_offset <= p_end_offset:
                    if len(cur_offset_list) == 0:
                        cur_offset_list = [b_idx+1]
                    else:
                        cur_offset_list.append(b_idx+1)

            p_result.bert_offset = cur_offset_list

        # The output type: Frozen word embeddings
        if args.output_type == "frozen":
            bert_embedding = model(input_ids=tokenized_result['input_ids'],
                                  attention_mask=tokenized_result['attention_mask'],
                                  token_type_ids=tokenized_result['token_type_ids'])[0]

            bert_embedding = torch.squeeze(bert_embedding, 0)
            return {'emb':bert_embedding,
                     'preprocessed_offset_match':[x.bert_offset for x in preprocessed_result],
                     'preprocessed_dep': [x.dep for x in preprocessed_result]}

        # The output type: BERT model input
        else:
            return {'input_ids':tokenized_result['input_ids'],
                 'attention_mask':tokenized_result['attention_mask'],
                 'token_type_ids':tokenized_result['token_type_ids'],
                 'preprocessed_offset_match':[x.bert_offset for x in preprocessed_result],
                 'preprocessed_dep': [x.dep for x in preprocessed_result]}


    elif args.emb =="elmo":
        elmo_embedding = embedder.embed_sentence([x.tk for x in preprocessed_result])[0]
        elmo_embedding = torch.tensor(elmo_embedding)

        return {'emb':elmo_embedding,
                 'preprocessed_offset_match':[[x.tk_idx] for x in preprocessed_result],
                 'preprocessed_dep': [x.dep for x in preprocessed_result]}

    elif 'glove' in args.emb:
        glove_emb = []

        for x in preprocessed_result:
            cur_tk = x.tk.lower()
            if cur_tk in embedder:
                cur_emb = embedder[cur_tk]
            else:
                cur_emb = np.zeros(300)
            glove_emb.append(cur_emb)
        glove_emb = torch.tensor(glove_emb)

        return {'emb':glove_emb,
                 'preprocessed_offset_match':[[x.tk_idx] for x in preprocessed_result],
                 'preprocessed_dep': [x.dep for x in preprocessed_result]}
