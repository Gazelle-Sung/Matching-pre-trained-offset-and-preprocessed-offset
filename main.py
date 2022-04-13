import argparse
import PreTrained2PreProcessed as pt2pp


def main(args):
    sample_text = """Officials are set to announce details of B.C.'s latest restart plan on Tuesday as daily case counts continue to trend downward and hours after the last round of "circuit breaker" restrictions expired."""
    model, embedder = pt2pp.sel_pretrained(args.emb)

    result = pt2pp.embedding2preprocessed(args, sample_text, embedder, model)

    if args.emb == "bert" and args.output_type != "frozen":
        print(result['input_ids'])
        print(result['attention_mask'])
        print(result['token_type_ids'])
        print(result['preprocessed_offset_match'])
        print(result['preprocessed_dep'])
    else:
        print(result['emb'])
        print(result['preprocessed_offset_match'])
        print(result['preprocessed_dep'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pipeline", default="spacy", type=str, help="Preprocessing method such as Stanza or Spacy.")
    parser.add_argument("--emb", default="bert", type=str, help="Pre-trained word embedding such as BERT, ELMo or Glove.")
    parser.add_argument("--output_type", default="frozen", type=str, help="The output type of either frozen word embedding or BERT input for BERT embedding.")

    args = parser.parse_args()

    main(args)
