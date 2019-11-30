
'''
Here start to convert the above entailment samples into persona file
'''
# Map the labels to the dialogue

counter_dnli, counter_tnli, counter_both = 0, 0, 0

for fn in ["valid", "test", "train"]:
    fr_dnli = open("../data/ConvAI2/{}_both_original_dialogueNLI.txt".format(fn), "r")
    fr_tnli = open("../data/ConvAI2/{}_both_original_BERT.txt".format(fn), "r")
    fr_tfidf = open("../data/ConvAI2/{}_both_original_tfidf.txt".format(fn), "r")
    fw = open("../data/ConvAI2/{}_both_original_final.txt".format(fn), "w")

    lines_dnli = fr_dnli.readlines()
    lines_tnli = fr_tnli.readlines()
    lines_tfidf = fr_tfidf.readlines()

    assert len(lines_dnli)==len(lines_tnli)
    assert len(lines_dnli)==len(lines_tfidf)

    for line_idx in range(len(lines_dnli)): 
        line_dnli = lines_dnli[line_idx]
        line_tnli = lines_tnli[line_idx]
        line_tfidf = lines_tfidf[line_idx]
        if "partner's persona: " in line_dnli or "your persona: " in line_dnli:
            # assert line_dnli==line_tnli
            # assert line_dnli==line_tfidf
            fw.write(line_dnli)
        else:
            num, line_dnli  = lines_dnli[line_idx].split("\t", 1)
            num, line_tnli  = lines_tnli[line_idx].split("\t", 1)
            num, line_tfidf = lines_tfidf[line_idx].split("\t", 1)
            sentence = line_dnli.split("\t")[0].replace("\n", "")

            if line_dnli==line_tnli and line_tnli==line_tfidf:
                fw.write(num+"\t"+line_dnli)
            else:
                combine_set = set()
                line_dnli_set  = set(line_dnli.replace("\n", "").split("\t")[1:]) if len(line_dnli.split("\t"))>1 else set()
                line_tnli_set  = set(line_tnli.replace("\n", "").split("\t")[1:]) if len(line_tnli.split("\t"))>1 else set()
                line_tfidf_set = set(line_tfidf.replace("\n", "").split("\t")[1:]) if len(line_tfidf.split("\t"))>1 else set()

                combine_set.update(line_dnli_set)
                combine_set.update(line_tnli_set)
                combine_set.update(line_tfidf_set)

                new_line = [num, sentence] + [s for s in combine_set if (s!="None" and s!="['<none>', '<none>', '<none>']")]
                fw.write("\t".join(new_line)+"\n")

                # Statistics 
                # if combine_set == line_dnli_set and combine_set != line_tnli_set:
                #     counter_dnli += 1
                # if combine_set == line_tnli_set and combine_set != line_dnli_set:
                #     counter_tnli += 1
                # if combine_set != line_dnli_set and combine_set != line_tnli_set:
                #     counter_both += 1

# print("counter_dnli {}, counter_tnli {}, counter_both {}".format(counter_dnli, counter_tnli, counter_both))
