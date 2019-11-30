from allennlp.predictors.predictor import Predictor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from tqdm import tqdm
import numpy as np
from load_bert import bert_model
import jsonlines
import string

'''
HOW TO RUN:
CUDA_VISIBLE_DEVICES=1 python ent_mapping.py --data_dir data/dialogue_nli/ --task_name persona --bert_model bert-base-uncased --output_dir data/nli_model/ --do_eval
'''


'''
Here get the sent2triple dictioinary
'''
def remove_punctuation(x):
    x = "".join([c for c in x if c not in string.punctuation])
    x = [s for s in x.split() if s]
    x = " ".join(x)
    return x

bert = bert_model()

## TEST
# bert.predict_label(["hello , how are you doing tonight ?"
#                     ,"hello , how are you doing tonight ?"
#                     ,"hello , how are you doing tonight ?"
#                     ,"hello , how are you doing tonight ?"
#                     ,"hello , how are you doing tonight ?"
#                     ],
#                     ["i just bought a brand new house ."
#                     ,"i like to dance at the club ."
#                     ,"i run a dog obedience school ."
#                     ,"i have a big sweet tooth ."
#                     ,"i like taking and posting selkies ."])
# Get only the positive samples from the data
s2t_map = {}
triple_set = set()
for fn in ["dev", "test", "train"]:
    data = jsonlines.Reader(open('data/dialogue_nli/dialogue_nli_{}.jsonl'.format(fn), 'r')).read()
    for d in data:
        s2t_map[remove_punctuation(d["sentence1"])] = str(d["triple1"])
        s2t_map[remove_punctuation(d["sentence2"])] = str(d["triple2"])
        triple_set.add(str(d["triple1"]))
        triple_set.add(str(d["triple2"]))

print("Number of sentence triple pairs: ", len(s2t_map))
print("Number of triplet set: ", len(triple_set))

both_persona = []
file_name = ["valid", "test", "train"]

for fn in file_name:
    fr = open("data/ConvAI2/{}_both_original.txt".format(fn), "r")
    lines = fr.readlines()

    for line in lines: 
        if "partner's persona: " in line:
            persona = line.split("partner's persona: ")[1]
            persona = " ".join(nltk.word_tokenize(persona))
            both_persona.append(persona)
            # print(persona)
        elif "your persona: " in line:
            persona = line.split("your persona: ")[1]
            persona = " ".join(nltk.word_tokenize(persona))
            both_persona.append(persona)
    fr.close()

# def entailtment_score(turn, personas_items, predictor):
#     num_to_ent = ["Entailment","Contradiction","Neutral"]
#     p_res = []
#     value = []
#     for p in personas_items:
#         res = predictor.predict(
#             hypothesis=p,
#             premise=turn
#         )
#         ris = num_to_ent[np.argmax(res["label_probs"])] 
#         if(ris == "Entailment"):
#             p_res.append(p)
#             value.append(res["label_probs"][0])
#         else:
#             p_res.append("None")
#             value.append(0.0)

#     return p_res[np.argmax(value)]

# predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz")
# predictor._model = predictor._model.cuda()      

for fn in file_name:
    fr = open("data/ConvAI2/{}_both_original.txt".format(fn), "r")
    fw = open("data/ConvAI2/{}_both_original_BERT.txt".format(fn), "w")
    lines = fr.readlines()

    bar = tqdm(lines)
    for line in bar: 
        num, line = line.split(" ", 1)
        if int(num) == 1: # New Dialogue
            partner_parsona = []
            your_persona = []
            partner_parsona_triple = []
            your_persona_triple = []
            counter = 1

        if "partner's persona: " in line:
            persona = line.split("partner's persona: ")[1]
            persona = " ".join(nltk.word_tokenize(persona))
            partner_parsona.append(persona)
            try:
                persona_triple = s2t_map[remove_punctuation(persona)]
            except:
                persona_triple = "None"
            partner_parsona_triple.append(persona_triple)
            fw.write(line.replace("\n", "")+"\t"+str(persona_triple)+"\n")
            # fw.write(line)
            # print(persona)
        elif "your persona: " in line:
            persona = line.split("your persona: ")[1]
            persona = " ".join(nltk.word_tokenize(persona))
            your_persona.append(persona)
            try:
                persona_triple = s2t_map[remove_punctuation(persona)]
            except:
                persona_triple = "None"
            your_persona_triple.append(persona_triple)
            fw.write(line.replace("\n", "")+"\t"+str(persona_triple)+"\n")
            # fw.write(line)
            # print(persona)
        else:
            # partner_uttr = " ".join(line.split("\t")[0].split()[1:])
            partner_uttr = line.split("\t")[0]
            your_persona_uttr = line.split("\t")[1]

            imply_persona_partner = bert.predict_label([ partner_uttr for _ in range(len(partner_parsona))], partner_parsona, partner_parsona_triple)
            imply_persona_your = bert.predict_label([ your_persona_uttr for _ in range(len(your_persona))], your_persona, your_persona_triple)

            fw.write(str(counter)+"\t"+partner_uttr)
            for p in imply_persona_partner:
                fw.write("\t"+p)
            fw.write("\n")

            fw.write(str(counter+1)+"\t"+your_persona_uttr)
            for p in imply_persona_your:
                fw.write("\t"+p)
            fw.write("\n")

            counter += 2
            # write_line = line.split("\t")[:2] + [imply_persona_partner] + [imply_persona_your]
            # fw.write("\t".join(write_line))
            # fw.write("\n")

    fr.close()
    fw.close()



