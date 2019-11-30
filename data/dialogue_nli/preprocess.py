import jsonlines
from collections import Counter
from pprint import pprint


def preproc(kind):
    data = jsonlines.Reader(open('dialogue_nli_%s.jsonl' % kind, 'r')).read()
    mapper = {"negative":"contradiction", "positive":"entailment", "neutral":"neutral"}
    fw = open("{}.tsv".format(kind), "w")
    for i, d in enumerate(data):
        fw.write("{}\t{}\t{}\t{}\n".format(i, d["sentence1"],d["sentence2"],mapper[d['label']]))
    fw.close()

if __name__ == '__main__':
    preproc('train')
    preproc('dev')
    preproc('test')
