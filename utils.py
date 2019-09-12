import pickle

def read_pickle(filename):
    infile = open(filename, 'rb')
    data = pickle.load(infile)
    infile.close()
    return data


def dump_pickle(filename, data):
    outfile = open(filename, "wb")
    pickle.dump(data, filename)
    outfile.close()