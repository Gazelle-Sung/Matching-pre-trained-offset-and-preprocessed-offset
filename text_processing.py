import stanza
import spacy

# *** Tokenize with selected method and store with token-level id, text, offset, pos tag, ner and dependency triple
def tokenization_processing(_datasample, _pipeline='stanza'):
    cur_clause_feature = []
    
    # Spacy
    if _pipeline == "spacy":
        nlp = spacy.load("en_core_web_sm")
        print("****** Selected preprocessing method: {}".format(_pipeline))
        doc = nlp(_datasample)
        for tk_idx, tk in enumerate(doc):
            cur_tk = tk.text
            cur_id = tk_idx+1
            cur_head = tk.head.i

            cur_tk_start_offset = tk.idx
            cur_tk_end_offset = tk.idx + len(tk)

            cur_pos = tk.pos_
            cur_ner = tk.ent_type_
            cur_dep = tk.dep_

            cur_dep_triple = (cur_id, cur_dep, cur_head)

            cur_clause_feature.append(Clause_feature(cur_id, cur_tk, (cur_tk_start_offset, cur_tk_end_offset), cur_pos, cur_ner, cur_dep_triple, ''))
    # Stanza
    elif _pipeline == "stanza":
        nlp = stanza.Pipeline('en')
        print("****** Selected preprocessing method: {}".format(_pipeline))
        doc = nlp(_datasample)
        for sen in doc.sentences:
            for tk in sen.tokens:
                tk_infor_dict = tk.to_dict()[0]
                cur_tk = tk_infor_dict["text"]
                cur_id = tk_infor_dict['id']
                cur_head = tk_infor_dict['head']

                offsets = tk_infor_dict['misc'].split("|")
                cur_tk_start_offset = int(offsets[0].split("=")[1])
                cur_tk_end_offset = int(offsets[1].split("=")[1])

                cur_pos = tk_infor_dict["xpos"]
                cur_ner = tk_infor_dict["ner"]

                cur_dep = tk_infor_dict["deprel"]

                cur_dep_triple = (cur_id, cur_dep, cur_head)

                cur_clause_feature.append(Clause_feature(cur_id, cur_tk, (cur_tk_start_offset, cur_tk_end_offset), cur_pos, cur_ner, cur_dep_triple, ''))
    return cur_clause_feature
