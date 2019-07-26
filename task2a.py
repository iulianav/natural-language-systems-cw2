from nltk import pos_tag
from nltk.corpus import PlaintextCorpusReader, wordnet


# Read positive sentiment seed adjectives from file and save them in a list.
with open("positive_seeds.txt", "r") as f:
    text1 = f.read()
    pos_seeds = text1.split(" ")

# Read negative sentiment seed adjectives from file and save them in a list.
with open("negative_seeds.txt", "r") as f:
    text2 = f.read()
    neg_seeds = text2.split(" ")

# Read the reviews corpus and POS tag it using the NLTK pos tagger.
wordlists = PlaintextCorpusReader("rt-polaritydata", ".*")
POS_tagging = pos_tag(wordlists.words())

# POS tags associated with adjectives.
adjectives_tags = ["JJ", "JJR", "JJS"]

# Set containing all the unique negative adjectives.
negatives = set(neg_seeds)

# Set containing all te unique positive adjectives.
positives = set(pos_seeds)

# For each sampled adjective, store in a dictionary how many times it is assigned a positive sentiment
# and how many times it is assigned a negative sentiment.
sentiments_counts = {}

# Scan the corpus once to sample new adjectives and populate the lexicon.
for i, el in enumerate(POS_tagging):
    if el[1] in adjectives_tags and el[0] not in positives and el[0] not in negatives:

        if el[0] not in sentiments_counts:
            # Initialise sentiment counts for the current adjective.
            sentiments_counts[el[0]] = {
            "pos": 0,
            "neg": 0,
            }

        # Assign sentiments to the new adjectives based on the following patterns.
        if POS_tagging[i+1][0] == "and" and POS_tagging[i+2][1] in adjectives_tags:
            if POS_tagging[i+2][0] in positives:
                sentiments_counts[el[0]]["pos"] += 1    
            if POS_tagging[i+2][0] in negatives:
                sentiments_counts[el[0]]["neg"] += 1
        if POS_tagging[i+1][0] == "and" and POS_tagging[i+3][1] in adjectives_tags:
            if POS_tagging[i+3][0] in positives:
                sentiments_counts[el[0]]["pos"] += 1    
            if POS_tagging[i+3][0] in negatives:
                sentiments_counts[el[0]]["neg"] += 1
        if POS_tagging[i-1][0] == "and" and POS_tagging[i-2][1] in adjectives_tags:
            if POS_tagging[i-2][0] in positives:
                sentiments_counts[el[0]]["pos"] += 1
            if POS_tagging[i-2][0] in negatives:
                sentiments_counts[el[0]]["neg"] += 1
        if POS_tagging[i-1][0] == "and" and POS_tagging[i-3][1] in adjectives_tags:
            if POS_tagging[i-3][0] in positives:
                sentiments_counts[el[0]]["pos"] += 1
            if POS_tagging[i-3][0] in negatives:
                sentiments_counts[el[0]]["neg"] += 1
        if POS_tagging[i+1][0] == "but" and POS_tagging[i+2][1] in adjectives_tags:
            if POS_tagging[i+2][0] in positives:
                sentiments_counts[el[0]]["neg"] += 1
            if POS_tagging[i+2][0] in negatives:
                sentiments_counts[el[0]]["pos"] += 1
        if POS_tagging[i+1][0] == "but" and POS_tagging[i+3][1] in adjectives_tags:
            if POS_tagging[i+3][0] in positives:
                sentiments_counts[el[0]]["neg"] += 1
            if POS_tagging[i+3][0] in negatives:
                sentiments_counts[el[0]]["pos"] += 1
        if POS_tagging[i-1][0] == "but" and POS_tagging[i-2][1] in adjectives_tags:
            if POS_tagging[i-2][0] in positives:
                sentiments_counts[el[0]]["neg"] += 1
            if POS_tagging[i-2][0] in negatives:
                sentiments_counts[el[0]]["pos"] += 1
        if POS_tagging[i-1][0] == "but" and POS_tagging[i-3][1] in adjectives_tags:
            if POS_tagging[i-3][0] in positives:
                sentiments_counts[el[0]]["neg"] += 1
            if POS_tagging[i-3][0] in negatives:
                sentiments_counts[el[0]]["pos"] += 1
        if POS_tagging[i+1][0] == "yet" and POS_tagging[i+2][1] in adjectives_tags:
            if POS_tagging[i+2][0] in positives:
                sentiments_counts[el[0]]["neg"] += 1
            if POS_tagging[i+2][0] in negatives:
                sentiments_counts[el[0]]["pos"] += 1
        if POS_tagging[i+1][0] == "yet" and POS_tagging[i+3][1] in adjectives_tags:
            if POS_tagging[i+3][0] in positives:
                sentiments_counts[el[0]]["neg"] += 1
            if POS_tagging[i+3][0] in negatives:
                sentiments_counts[el[0]]["pos"] += 1
        if POS_tagging[i-1][0] == "yet" and POS_tagging[i-2][1] in adjectives_tags:
            if POS_tagging[i-2][0] in positives:
                sentiments_counts[el[0]]["neg"] += 1
            if POS_tagging[i-2][0] in negatives:
                sentiments_counts[el[0]]["pos"] += 1
        if POS_tagging[i-1][0] == "yet" and POS_tagging[i-3][1] in adjectives_tags:
            if POS_tagging[i-3][0] in positives:
                sentiments_counts[el[0]]["neg"] += 1
            if POS_tagging[i-3][0] in negatives:
                sentiments_counts[el[0]]["pos"] += 1
        if POS_tagging[i+1][0] == "," and POS_tagging[i+2][1] in adjectives_tags:
            if POS_tagging[i+2][0] in positives:
                sentiments_counts[el[0]]["pos"] += 1
            if POS_tagging[i+2][0] in negatives:
                sentiments_counts[el[0]]["neg"] += 1
        if POS_tagging[i-1][0] == "," and POS_tagging[i-2][1] in adjectives_tags:
            if POS_tagging[i-2][0] in positives:
                sentiments_counts[el[0]]["pos"] += 1
            if POS_tagging[i-2][0] in negatives:
                sentiments_counts[el[0]]["neg"] += 1


# Assigne the sampled adjectives the sentiment with the higher count.
for el in sentiments_counts.keys():
    if sentiments_counts[el]["neg"] != 0 and sentiments_counts[el]["pos"] != 0:
        if sentiments_counts[el]["neg"] > sentiments_counts[el]["pos"]:
            negatives.add(el)
        elif sentiments_counts[el]["pos"] > sentiments_counts[el]["neg"]:
            positives.add(el)


# Set containing the unique synonyms of positive words.
pos_synonyms = set()

# Set containing the unique antonyms of positive words.
pos_antonyms = set()

# Set containing the unique synonyms of negative words.
neg_synonyms = set()

# Set containing the unique antonyms of negative words.
neg_antonyms = set()


# Add the synonyms of positive adjectives found so far to positives in the lexicon
# and the antonyms of positive adjectives found so far to negatives in the lexicon.
for el in positives:
    for syn in wordnet.synsets(el):
        if syn.pos() in ["a", "s"]:
            for l in syn.lemmas():
                pos_synonyms.add(l.name())
                if l.antonyms():
                    for a in l.antonyms():
                        pos_antonyms.add(a.name())

# Add the synonyms of negative adjectives found so far to negatives in the lexicon
# and the antonyms of negative adjectives found so far to positives in the lexicon.
for el in negatives:
    for syn in wordnet.synsets(el):
        if syn.pos() in ["a", "s"]:
            for l in syn.lemmas():
                neg_synonyms.add(l.name())
                if l.antonyms():
                    for a in l.antonyms():
                        neg_antonyms.add(a.name())

# Add to the positives set the sampled positive sentiment adjectives based on the patterns, positves synonyms and negatives antonyms.
positives = positives | pos_synonyms | neg_antonyms

# Add to the negatives set the sampled negative sentiment adjectives based on the patterns, negatives synonyms and  positives antonyms.
negatives = negatives | neg_synonyms | pos_antonyms

with open("enriched_lexicon_positives.txt", "w") as f:
    for el in positives:
        f.write(el+"\n")

with open("enriched_lexicon_negatives.txt", "w") as f:
    for el in negatives:
        f.write(el+"\n")
