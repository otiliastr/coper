import argparse, logging
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Run edgelist creator')
    
    parser.add_argument('--input', nargs='?', default = '/zfsauton/home/gis/research/qa/models/prelim_tests/ConvE/data/nations/train.txt', help='Input trainfile')
    parser.add_argument('--mapper', nargs = '?', default = '/zfsauton/home/gis/.data/nations/vocab_e1', help='Input entity id mapper')
    parser.add_argument('--output', nargs = '?', default = '/zfsauton/home/gis/.data/nations/nations.edgelist', help='Output Edgelist')
    
    return parser.parse_args()


def read_mapper(mapfile):
    with open(mapfile, 'rb') as f:
        data = pickle.load(f)
    return data[0]

def read_data(datafile):
    #print(datafile)
    with open(datafile, 'r') as f:
        data = f.readlines()
    return data

def gen_edgelist(mapper, data, outputfile):
    with open(outputfile, 'a+') as out_file:
        for line in data:
            e1, _, e2 = line.split("\t")
            e1_id = mapper[e1.strip()]
            e2_id = mapper[e2.strip()]
            out_line = str(e1_id) + " " + str(e2_id) + "\n"
            out_file.write(out_line)

def main(args):
    input_data = read_data(args.input)
    mapper = read_mapper(args.mapper)
    gen_edgelist(mapper, input_data, args.output)



if __name__ == "__main__":
    args = parse_args()
    main(args)
