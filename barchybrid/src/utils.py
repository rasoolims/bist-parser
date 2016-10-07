from collections import Counter
import re

class ConllEntry:
    def __init__(self, id, form, pos, parent_id=None, relation=None, lang_id=None, weight= 1.0):
        self.id = id
        self.form = form
        self.norm = normalize(form)
        self.pos = pos.upper()
        self.parent_id = parent_id
        self.relation = relation
        self.lang_id = lang_id
        self.weight = weight

class ParseForest:
    def __init__(self, sentence):
        self.roots = list(sentence)

        for root in self.roots:
            root.children = []
            root.scores = None
            root.parent = None
            root.pred_parent_id = 0  # None
            root.pred_relation = 'rroot'  # None
            root.vecs = None
            root.lstms = None

    def __len__(self):
        return len(self.roots)

    def Attach(self, parent_index, child_index):
        parent = self.roots[parent_index]
        child = self.roots[child_index]

        child.pred_parent_id = parent.id
        del self.roots[child_index]

def isProj(sentence):
    forest = ParseForest(sentence)
    unassigned = {entry.id: sum([1 for pentry in sentence if pentry.parent_id == entry.id]) for entry in sentence}

    for _ in xrange(len(sentence)):
        for i in xrange(len(forest.roots) - 1):
            if forest.roots[i].parent_id == forest.roots[i + 1].id and unassigned[forest.roots[i].id] == 0:
                unassigned[forest.roots[i + 1].id] -= 1
                forest.Attach(i + 1, i)
                break
            if forest.roots[i + 1].parent_id == forest.roots[i].id and unassigned[forest.roots[i + 1].id] == 0:
                unassigned[forest.roots[i].id] -= 1
                forest.Attach(i, i + 1)
                break

    return len(forest.roots) == 1

def vocab(conll_path):
    wordsCount = Counter()
    posCount = Counter()
    relCount = Counter()
    langCount = Counter()

    with open(conll_path, 'r') as conllFP:
        for sentence in read_conll(conllFP, True):
            wordsCount.update([node.norm for node in sentence])
            posCount.update([node.pos for node in sentence])
            relCount.update([node.relation for node in sentence])
            langCount.update([node.lang_id for node in sentence])
    return (wordsCount, {w: i for i, w in enumerate(wordsCount.keys())}, posCount.keys(), relCount.keys(), langCount.keys())

def read_conll(fh, proj):
    dropped = 0
    read = 0
    last_lang_id = None
    last_weight = 1
    tokens = []
    for line in fh:
        tok = line.strip().split()
        if not tok:
            if len(tokens) > 1:
                root = ConllEntry(0, '*root*', 'ROOT-POS', 0, 'rroot', last_lang_id, last_weight)
                tokens = [root] + tokens
                if not proj or isProj(tokens):
                    yield tokens
                else:
                    # print 'Non-projective sentence dropped'
                    dropped += 1
                read += 1
            tokens = []
        else:
            last_lang_id = tok[5]
            weight = 1
            try:
                weight = int(tok[8])
            except:
                weight = 1
            last_weight = weight
            tokens.append(ConllEntry(int(tok[0]), tok[1], tok[3], int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[5], weight))
    if len(tokens) > 1:
        root = ConllEntry(0, '*root*', 'ROOT-POS', 0, 'rroot', last_lang_id, last_weight)
        tokens = [root] + tokens
        yield tokens

    print dropped, 'dropped non-projective sentences.'
    print read, 'sentences read.'

def write_conll(fn, conll_gen):
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                fh.write('\t'.join(
                    [str(entry.id), entry.form, '_', entry.pos, entry.pos, entry.lang_id, str(entry.pred_parent_id),
                     entry.pred_relation, '_', '_']))
                fh.write('\n')
            fh.write('\n')

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");

def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()