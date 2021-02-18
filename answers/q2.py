lang_count = {}
for file in ('train.en', 'train.de'):
    count = lang_count.setdefault(file, {})

    document = open(f'../europarl_raw/{file}', 'r').read()[:-1]
    for sentence in document.split('\n'):
        for word in sentence.split(' '):  # ??? what about punctuations?
            count[word] = count.setdefault(word, 0) + 1

# todo 1.How many word tokens are in the English data? In the German data? Give both the total count and the number of word types in each language.
print('\n\n1.')
for lang, count in lang_count.items():
    print(f'Number of {lang} word tokens: {sum(count.values())}')
    print(f'Number of {lang} word types: {len(count.keys())}')

print('\n\n2.')
# todo 2.How many word tokens will be replaced by <UNK> in English? In German? Sub- sequently, what will the total vocabulary size be?
lang_unk = {}
for lang, count in lang_count.items():
    lang_unk[lang] = [w for w, c in count.items() if c == 1]
    print(f'Number of <UNK> in {lang}: {len(lang_unk[lang])}')

print('\n\n3.')
# todo 3.Inspect the words which will be replaced by <UNK>. Is there a specific type of word which will be commonly replaced?
#  Give an example of this type of word from English and German. How will this affect the test set?
#  What ‘type’ could refer to here could be a specific POS tag or entity class, but you are free to come up with another idea based upon your understanding of the data.
for lang, unk in lang_unk.items():
    print(lang, unk)
    # ??? number? subject?

print('\n\n4.')
# todo 4.How many unique vocabulary tokens are the same between both languages? How could we exploit this similarity in our model?
#  You don’t have to consider false friends such as the English verb ‘die’ and German article ‘die’, just treat them as the same.
# ??? what does "unique" mean.
# word_types = list(lang_unk.values())
word_types = [count.keys() for count in lang_count.values()]
intersection = set(word_types[0]) & set(word_types[1])
print(f'Number of same unique tokens: {len(intersection)}')
print(intersection)

print('\n\n5.')
# todo 5.Given the observations above, how do you think the NMT system will be affected by differences in sentence length, token ratios, and unknown word handling?
# ???
