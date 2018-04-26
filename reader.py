
def file_reader(filename):
    def reader():
        for line in open(filename, 'r'):
            line = line.strip()
            features = line.split(";")
            word_idx = []
            for item in features[1].strip().split(" "):
                word_idx.append(int(item))
                target_idx = []
            for item in features[2].strip().split(" "):
                label_index = int(item)
                if label_index == 0:
                    label_index = 48
                else:
                    label_index -= 1   
                target_idx.append(label_index)
            mention_idx = []
            for item in features[3].strip().split(" "):
                mention_idx.append(int(item))
            yield word_idx, mention_idx, target_idx,
    return reader

def load_dict(dict_path):
    return dict((line.strip().split("\t")[0], idx)
                for idx, line in enumerate(open(dict_path, "r").readlines()))


def load_reverse_dict(dict_path):
    return dict((idx, line.strip().split("\t")[0])
                for idx, line in enumerate(open(dict_path, "r").readlines()))
