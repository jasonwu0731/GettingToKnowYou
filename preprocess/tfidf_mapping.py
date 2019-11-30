from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from tqdm import tqdm
import numpy as np
import jsonlines
import string

'''
Here get the sent2triple dictioinary
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
s2t_map = {}
triple_set = set()
for fn in ["dev", "test", "train"]:
    data = jsonlines.Reader(open('../data/dialogue_nli/dialogue_nli_{}.jsonl'.format(fn), 'r')).read()
    for d in data:
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

print("Number of sentence triple pairs: ", len(s2t_map))
print("Number of triplet set: ", len(triple_set))


'''
Here get the tfidf dictioinary
'''
both_persona = []
file_name = ["valid", "test", "train"]
for fn in file_name:
    fr = open("../data/ConvAI2/{}_both_original.txt".format(fn), "r")
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
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer = tfidf_vectorizer.fit(both_persona)  # finds the tfidf score with normalization

'''
Here start to convert the above entailment samples into persona file
'''
for fn in file_name:
    fr = open("../data/ConvAI2/{}_both_original.txt".format(fn), "r")
    fw = open("../data/ConvAI2/{}_both_original_tfidf.txt".format(fn), "w")
    lines = fr.readlines()
    bar = tqdm(lines)
    for line in bar: 
        num, line = line.split(" ", 1)
        if int(num) == 1: # New Dialogue
            partner_persona = []
            your_persona = []
            partner_persona_triple = []
            your_persona_triple = []
            counter = 1
        if "partner's persona: " in line:
            persona = line.split("partner's persona: ")[1]
            persona = " ".join(nltk.word_tokenize(persona))
            partner_persona.append(persona)
            try:
                persona_triple = remove_none(s2t_map[remove_punctuation(persona)])
            except:
                # Means this persona never appeal in the positive entailment label, we can skip
                persona_triple = []
            partner_persona_triple.append(persona_triple)
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
            partner_persona_vec = tfidf_vectorizer.transform(partner_persona)
            your_persona_vec = tfidf_vectorizer.transform(your_persona)
            partner_uttr = line.split("\t")[0]
            your_persona_uttr = line.split("\t")[1]
            partner_uttr_vec = tfidf_vectorizer.transform([partner_uttr])
            your_persona_uttr_vec = tfidf_vectorizer.transform([your_persona_uttr])
            score_partner = cosine_similarity(partner_uttr_vec, partner_persona_vec)
            score_your = cosine_similarity(your_persona_uttr_vec, your_persona_vec)

            # imply_persona_triple_partner = [partner_persona_triple[i] for i, s in enumerate(score_partner[0]) if s > 0.2]
            # imply_persona_triple_your = [your_persona_triple[i] for i, s in enumerate(score_your[0]) if s > 0.2]
            imply_persona_triple_partner = [partner_persona_triple[np.argmax(score_partner[0])]] if max(score_partner[0]) > 0.2 else []
            imply_persona_triple_your = [your_persona_triple[np.argmax(score_your[0])]] if max(score_your[0]) > 0.2 else []

            fw.write(str(counter)+"\t"+partner_uttr)
            for p in imply_persona_triple_partner:
                for pp in p:
                    fw.write("\t"+pp)
            fw.write("\n")

            fw.write(str(counter+1)+"\t"+your_persona_uttr)
            for p in imply_persona_triple_your:
                for pp in p:
                    fw.write("\t"+pp)
            fw.write("\n")

            counter += 2
            # if len(imply_persona_triple_partner)>1 or len(imply_persona_triple_your)>1: break

    fr.close()
    fw.close()



