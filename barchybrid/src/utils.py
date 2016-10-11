from collections import Counter
import re, codecs

class ConllStruct:
    def __init__(self, entries, predicates):
        self.entries = entries
        self.predicates = predicates

class ConllEntry:
    def __init__(self, id, form, lemma, pos, sense = None, parent_id=None, relation=None, predicateList=None):
        self.id = id
        self.form = form
        self.lemma = lemma
        self.norm = normalize(form)
        self.lemmaNorm = normalize(lemma)
        self.pos = pos.upper()
        self.parent_id = parent_id
        self.relation = relation
        self.predicateList = predicateList
        self.sense = sense

    def __str__(self):
        entry_list = [str(self.id), self.form, self.lemma, self.lemma, self.pos, self.pos, '_', '_',
                      str(self.pred_parent_id),
                      str(self.pred_parent_id), self.pred_relation, self.pred_relation,
                      '_' if self.sense == '_' else 'Y',
                      self.sense, '_']
        for p in self.predicateList.values():
            entry_list.append(p)
        return '\t'.join(entry_list)

def vocab(conll_path):
    wordsCount = Counter()
    lemma_count = Counter()
    posCount = Counter()
    relCount = Counter()
    semRelCount = Counter()

    with open(conll_path, 'r') as conllFP:
        for sentence in read_conll(conllFP):
            wordsCount.update([node.norm for node in sentence])
            lemma_count.update([node.lemma for node in sentence])
            posCount.update([node.pos for node in sentence])
            relCount.update([node.relation for node in sentence])
            for node in sentence:
                for pred in node.predicateList.values():
                    if pred!='_':
                        semRelCount.append(pred)


    return (wordsCount, {w: i for i, w in enumerate(wordsCount.keys())}, lemma_count, {w: i for i, w in enumerate(lemma_count.keys())}, posCount.keys(), relCount.keys(), semRelCount.keys())

def read_conll(fh):
    sentences = open(fh, 'r').read().strip().split('\n\n')
    read = 0
    for sentence in sentences:
        words = []
        words.append(ConllEntry(0, 'ROOT','ROOT', 'ROOT', 'ROOT'))
        predicates = list()
        entries = sentence.strip().split('\n')
        for entry in entries:
            spl = entry.split('\t')
            predicateList = dict()
            if spl[12]=='Y':
                predicates.append(int(spl[0]))

            for i in range(14, len(spl)):
                predicateList[i - 14] = spl[i]

            words.append(ConllEntry(int(spl[0]), spl[1], spl[3], spl[5], spl[13], int(spl[9]), spl[11], predicateList))
        read+=1
        yield  ConllStruct(words, predicates)
    print read, 'sentences read.'

def write_conll(fn, conll_structs):
    with codecs.open(fn, 'w') as fh:
        for conll_struct in conll_structs:
            for entry in conll_struct.entries:
                fh.write(entry)
                fh.write('\n')
            fh.write('\n')

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");

def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()