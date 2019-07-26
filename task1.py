from itertools import groupby

from nltk import pos_tag
from nltk.chunk import ne_chunk
from nltk.corpus import inaugural
from nltk.tag import StanfordNERTagger
from nltk.tree import Tree

# Uncomment to check the required StanfordNERTagger environment variables.
# print os.environ.get("CLASSPATH")
# print os.environ.get("STANFORD_MODELS")

# Read the corpus and POS tag it.
POS_tagging = pos_tag(inaugural.words())

# Process the corpus with the NLTK named entity classifier.
ne_nltk = ne_chunk(POS_tagging)

# Filter out in a list only the organization entities. Join by space words that are part of the same organization entity (same Tree object).
nltk_organizations = [" ".join(w[0] for w in el) for el in ne_nltk if (type(el) == Tree and el.label() == "ORGANIZATION")]

# Remove duplicates.
nltk_organizations = set(nltk_organizations)

# Filter out in a list only the person entities. Join by space words that are part of the same person entity (same Tree object).
nltk_persons = [" ".join(w[0] for w in el) for el in ne_nltk if (type(el) == Tree and el.label() == "PERSON")]

# Remove duplicates.
nltk_persons = set(nltk_persons)

print("\nTotal number of unique NLTK organizations: " + str(len(nltk_organizations)))
print("Total number of unique NLTK persons: " + str(len(nltk_persons)) + "\n")

# Instantiate the Stanford named entity recognizer and process the corpus.
st = StanfordNERTagger("english.all.3class.distsim.crf.ser.gz")
ne_st = st.tag(inaugural.words())

# Join by space neighbouring words that are part of the same entity.
processed_ne_st = []
for tag, chunk in groupby(ne_st, lambda x:x[1]):
    if tag != "O":
        processed_ne_st.append((" ".join(w for w, t in chunk), tag))

# Filter out in a list only the organization entities.
st_organizations = [el[0] for el in processed_ne_st if el[1] == "ORGANIZATION"]

# Remove duplicates.
st_organizations = set(st_organizations)

# Filter out in a list only the person entities.
st_persons = [el[0] for el in processed_ne_st if el[1] == "PERSON"]

# Remove duplicates.
st_persons = set(st_persons)

print("Total number of unique Stanford organizations: " + str(len(st_organizations)))
print("Total number of unique Stanford persons: " + str(len(st_persons)) + "\n")

# Count the number of exact matches for organization entities between NLTK and Stanford.
exact_match_organizations = st_organizations & nltk_organizations

# Count the number of partial matches for organization entities between NLTK and Stanford.
partial_match_organizations = 0
for i  in st_organizations:
	for j in nltk_organizations:
		if (i in j or j in i) and i != j:
			partial_match_organizations += 1

print("Number of organizations exact matches: " + str(len(exact_match_organizations)))
print("Number of organizations partial matches: " + str(partial_match_organizations) + "\n")

# Count the number of exact matches for person entities between NLTK and Stanford.
exact_match_persons = st_persons & nltk_persons

# Count the number of partial matches for person entities between NLTK and Stanford.
partial_match_persons = 0
for i  in st_persons:
	for j in nltk_persons:
		if (i in j or j in i) and i != j:
			partial_match_persons += 1

print("Number of persons exact matches: " + str(len(exact_match_persons)))
print("Number of persons partial matches: " + str(partial_match_persons) + "\n")

# Write the identified entities to specific files, one per line.

with open("nltk_organizations.txt", "w") as f:
	for el in nltk_organizations:
		f.write(el + "\n")

with open("nltk_persons.txt", "w") as f:
	for el in nltk_persons:
		f.write(el + "\n")

with open("st_organizations.txt", "w") as f:
	for el in st_organizations:
		f.write(el + "\n")

with open("st_persons.txt", "w") as f:
	for el in st_persons:
		f.write(el + "\n")
