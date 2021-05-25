import os
import pickle
import numpy as np

def make_dict(_Lines):
    glove_dict = {}
    for line in _Lines[:-1]:
        token = line.split()[:-300]
        token = " ".join(token)
        we = line.split()[-300:]
        we = np.array(we).astype(np.float)
        glove_dict[token] = we
    return glove_dict

def write_pkl(_dict, _write_file_name):
    with open(write_file_name, 'wb') as f:
        pickle.dump(_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

def main(args):
    glove_name = 'glove.'+args.selected_glove+'.300d'

    # Read Glove Word embedding line by line
    file = open(glove_name+'.txt', 'r', encoding="utf-8")
    Lines = file.readlines()

    # make it dictionary
    glove_dict = make_dict(Lines)

    # Write pickle
    write_file_name = glove_name+'.pkl'
    write_pkl(glove_dict, write_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # select glove model either 42B or 840B
    parser.add_argument("--selected_glove", default="42B", type=str, help="Selected glove pre-trained word embeddings.")

    args = parser.parse_args()

    main(args)
