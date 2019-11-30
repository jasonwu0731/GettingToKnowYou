import nltk
from tqdm import tqdm
import numpy as np
import jsonlines
import string

'''
Here get the sent2triple dictioinary and the labeled NLI dataset
'''

def remove_punctuation(x):
    x = "".join([c for c in x if c not in string.punctuation])
    x = [s for s in x.split() if s]
    x = " ".join(x)
    return x

def remove_none(belief_arr):
    belief_arr = [b for b in belief_arr if b not in ["['<none>', '<none>', '<none>']", "['', '', '']"] ]
    return belief_arr

# Get only the positive samples from the data
entail_data_all = []
entail_data_sent_pair = []
s2t_map = {}
triple_set = set()
for fn in ["dev", "test", "train"]:
    data = jsonlines.Reader(open('../data/dialogue_nli/dialogue_nli_{}.jsonl'.format(fn), 'r')).read()
    entail_data = []
    pbar = tqdm(data)
    for d in pbar:
        if d["dtype"] in ["matchingtriple_pp", "matchingtriple_up"]:
            d["sentence1"] = remove_punctuation(d["sentence1"])
            d["sentence2"] = remove_punctuation(d["sentence2"])
            entail_data.append(d)
        # s2t_map[remove_punctuation(d["sentence1"])] = str(d["triple1"])
        # s2t_map[remove_punctuation(d["sentence2"])] = str(d["triple2"])
        if remove_punctuation(d["sentence1"]) not in s2t_map.keys():
            s2t_map[remove_punctuation(d["sentence1"])] = []
        if str(d["triple1"]) not in s2t_map[remove_punctuation(d["sentence1"])]:
            s2t_map[remove_punctuation(d["sentence1"])].append(str(d["triple1"]))
        if remove_punctuation(d["sentence2"]) not in s2t_map.keys():
            s2t_map[remove_punctuation(d["sentence2"])] = []
        if str(d["triple2"]) not in s2t_map[remove_punctuation(d["sentence2"])]:
            s2t_map[remove_punctuation(d["sentence2"])].append(str(d["triple2"]))
        triple_set.add(str(d["triple1"]))
        triple_set.add(str(d["triple2"]))

    entail_data_sent_pair += [ set([d["sentence1"], d["sentence2"]]) for d in entail_data]
    entail_data_all += entail_data

# Use dialogue_nli extra data
for fn in ["train", "dev", "test"]:
    data = jsonlines.Reader(open('../data/dialogue_nli_extra/dialogue_nli_EXTRA_uu_{}.jsonl'.format(fn), 'r')).read()
    pbar = tqdm(data)
    for d in pbar:
        try:
            if remove_punctuation(d["sentence1"]) not in s2t_map.keys():
                s2t_map[remove_punctuation(d["sentence1"])] = []
            if str(d["triple1"]) not in s2t_map[remove_punctuation(d["sentence1"])]:
                s2t_map[remove_punctuation(d["sentence1"])].append(str(d["triple1"]))
            
            if remove_punctuation(d["sentence2"]) not in s2t_map.keys():
                s2t_map[remove_punctuation(d["sentence2"])] = []
            if str(d["triple2"]) not in s2t_map[remove_punctuation(d["sentence2"])]:
                s2t_map[remove_punctuation(d["sentence2"])].append(str(d["triple2"]))

            triple_set.add(str(d["triple1"]))
            triple_set.add(str(d["triple2"]))
        except:
            continue

print("Number of positive entailment samples: ", len(entail_data_all))
print("Number of sentence triple pairs: ", len(s2t_map))
print("Number of triplet set: ", len(triple_set))


'''
Here start to convert the above entailment samples into persona file
'''
# Map the labels to the dialogue
for fn in ["valid", "test", "train"]:
    fr = open("../data/ConvAI2/{}_both_original.txt".format(fn), "r")
    fw = open("../data/ConvAI2/{}_both_original_dialogueNLI.txt".format(fn), "w")
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
                persona_triple = remove_none(s2t_map[remove_punctuation(persona)])
            except:
                # Means this persona never appeal in the positive entailment label, we can skip
                persona_triple = []
            partner_parsona_triple.append(persona_triple)
            fw.write(line.replace("\n", ""))
            for p in persona_triple:
                fw.write("\t"+p)
            fw.write("\n")
        elif "your persona: " in line:
            persona = line.split("your persona: ")[1]
            persona = " ".join(nltk.word_tokenize(persona))
            your_persona.append(persona)
            try:
                persona_triple = remove_none(s2t_map[remove_punctuation(persona)])
            except:
                # Means this persona never appeal in the positive entailment label, we can skip
                persona_triple = []
            your_persona_triple.append(persona_triple)
            fw.write(line.replace("\n", ""))
            for p in persona_triple:
                fw.write("\t"+p)
            fw.write("\n")
        else:
            partner_uttr = line.split("\t")[0]
            your_persona_uttr = line.split("\t")[1]

            imply_persona_triple_partner = set()
            for ip, p in enumerate(partner_parsona):
                if set([remove_punctuation(partner_uttr), remove_punctuation(p)]) in entail_data_sent_pair or \
                    set([remove_punctuation(p), remove_punctuation(partner_uttr)]) in entail_data_sent_pair:
                    for pt in partner_parsona_triple[ip]:
                        imply_persona_triple_partner.add(pt)
            try:
                sent_know = remove_none(s2t_map[remove_punctuation(partner_uttr)])
            except:
                sent_know = []
            for sk in sent_know:
                imply_persona_triple_partner.add(sk)

            imply_persona_triple_your = set()
            for iy, p in enumerate(your_persona):
                if set([remove_punctuation(your_persona_uttr), remove_punctuation(p)]) in entail_data_sent_pair or \
                    set([remove_punctuation(p), remove_punctuation(your_persona_uttr)]) in entail_data_sent_pair:
                    for yt in your_persona_triple[iy]:
                        imply_persona_triple_your.add(yt)
            try:
                sent_know = remove_none(s2t_map[remove_punctuation(your_persona_uttr)])
            except:
                sent_know = []
            for sk in sent_know:
                imply_persona_triple_your.add(sk)

            fw.write(str(counter)+"\t"+partner_uttr)
            for p in imply_persona_triple_partner:
                fw.write("\t"+p)
            fw.write("\n")

            fw.write(str(counter+1)+"\t"+your_persona_uttr)
            for p in imply_persona_triple_your:
                fw.write("\t"+p)
            fw.write("\n")

            counter += 2

    fr.close()
    fw.close()



