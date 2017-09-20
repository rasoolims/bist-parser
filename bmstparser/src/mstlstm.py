from dynet import *
from utils import read_conll, write_conll
from operator import itemgetter
import utils, time, random, decoder
import numpy as np


class MSTParserLSTM:
    def __init__(self, vocab, pos, rels, w2i, options):
        self.model = Model()
        random.seed(1)
        self.trainer = AdamTrainer(self.model, options.lr, options.beta1, options.beta2)
        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}
        self.activation = self.activations[options.activation]

        self.costaugFlag = options.costaugFlag
        self.dropout = False if options.dropout==0.0 else True
        self.dropout_prob = options.dropout

        self.ldims = options.lstm_dims
        self.wdims = options.we
        self.pdims = options.pe
        self.rdims = options.re
        self.layers = options.layer
        self.wordsCount = vocab
        self.vocab = {word: ind+3 for word, ind in w2i.iteritems()}
        self.pos = {word: ind+3 for ind, word in enumerate(pos)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels

        self.external_embedding, self.edim = None, 0
        if options.external_embedding is not None:
            external_embedding_fp = open(options.external_embedding,'r')
            external_embedding_fp.readline()
            self.external_embedding = {line.split(' ')[0] : [float(f) for f in line.strip().split(' ')[1:]] for line in external_embedding_fp}
            external_embedding_fp.close()

            self.edim = len(self.external_embedding.values()[0])
            self.noextrn = [0.0 for _ in xrange(self.edim)]
            self.extrnd = {word: i + 3 for i, word in enumerate(self.external_embedding)}
            self.elookup = self.model.add_lookup_parameters((len(self.external_embedding) + 3, self.edim))
            for word, i in self.extrnd.iteritems():
                self.elookup.init_row(i, self.external_embedding[word])
            self.extrnd['*PAD*'] = 1
            self.extrnd['*INITIAL*'] = 2

            print 'Load external embedding. Vector dimensions', self.edim

        self.deep_lstms = BiRNNBuilder(self.layers, 2*self.wdims + self.pdims + self.edim+1, self.ldims*2, self.model, VanillaLSTMBuilder)
        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units

        self.vocab['*PAD*'] = 1
        self.pos['*PAD*'] = 1

        self.vocab['*INITIAL*'] = 2
        self.pos['*INITIAL*'] = 2

        self.wlookup = self.model.add_lookup_parameters((len(vocab) + 3, self.wdims))
        self.hlookup = self.model.add_lookup_parameters((len(vocab) + 3, self.wdims))
        self.plookup = self.model.add_lookup_parameters((len(pos) + 3, self.pdims))
        self.rlookup = self.model.add_lookup_parameters((len(rels), self.rdims))

        self.hidLayerFOH = self.model.add_parameters((self.hidden_units, self.ldims * 2))
        self.hidLayerFOM = self.model.add_parameters((self.hidden_units, self.ldims * 2))
        self.hidBias = self.model.add_parameters((self.hidden_units))

        self.hid2Layer = self.model.add_parameters((self.hidden2_units, self.hidden_units))
        self.hid2Bias = self.model.add_parameters((self.hidden2_units))

        self.outLayer = self.model.add_parameters((1, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))

        self.rhidLayerFOH = self.model.add_parameters((self.hidden_units, 2 * self.ldims))
        self.rhidLayerFOM = self.model.add_parameters((self.hidden_units, 2 * self.ldims))
        self.rhidBias = self.model.add_parameters((self.hidden_units))

        self.rhid2Layer = self.model.add_parameters((self.hidden2_units, self.hidden_units))
        self.rhid2Bias = self.model.add_parameters((self.hidden2_units))

        self.routLayer = self.model.add_parameters((len(self.irels), self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))
        self.routBias = self.model.add_parameters((len(self.irels)))

    def  __getExpr(self, lstm_vecs, i, j):
        inp = self.hidLayerFOH.expr() * lstm_vecs[i][i] + self.hidLayerFOM.expr() * lstm_vecs[i][j]
        if self.hidden2_units > 0:
            output = self.outLayer.expr() * self.activation(self.hid2Bias.expr() + self.hid2Layer.expr() * self.activation(inp + self.hidBias.expr())) # + self.outBias
        else:
            output = self.outLayer.expr() * self.activation(inp + self.hidBias.expr()) # + self.outBias

        return output

    def __evaluate(self, lstm_vecs):
        exprs = [ [self.__getExpr(lstm_vecs, i, j) for j in xrange(len(lstm_vecs))] for i in xrange(len(lstm_vecs)) ]
        # scores = np.array([ [output.scalar_value() for output in exprsRow] for exprsRow in exprs ])
        return exprs

    def __evaluateLabel(self, lstm_vecs, i, j):
        inp_ =  self.rhidLayerFOH.expr() * lstm_vecs[i][i] + self.rhidLayerFOM.expr() *lstm_vecs[i][j]
        if self.hidden2_units > 0:
            output = self.routLayer.expr() * self.activation(self.rhid2Bias.expr() + self.rhid2Layer.expr() * self.activation(inp_+ self.rhidBias.expr())) + self.routBias.expr()
        else:
            output = self.routLayer.expr() * self.activation(inp_+ self.rhidBias.expr()) + self.routBias.expr()
        return output

    def Save(self, filename):
        self.model.save(filename)

    def Load(self, filename):
        self.model.populate(filename)

    def Predict(self, conll_path):
        self.deep_lstms.disable_dropout()
        with open(conll_path, 'r') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP)):
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                head_vec = [self.hlookup[int(self.vocab.get(entry.norm, 0))] if self.wdims > 0 else None for entry in conll_sentence]
                for entry in conll_sentence:
                    wordvec = self.wlookup[int(self.vocab.get(entry.norm, 0))] if self.wdims > 0 else None
                    posvec = self.plookup[int(self.pos[entry.pos])] if self.pdims > 0 else None
                    evec = self.elookup[int(self.extrnd.get(entry.form, self.extrnd.get(entry.norm, 0)))] if self.external_embedding is not None else None
                    entry.vec = concatenate(filter(None, [wordvec, posvec, evec]))

                lstm_vecs = list()
                for i in range(len(conll_sentence)):
                    indicator = [scalarInput(1) if j ==i else scalarInput(0) for j in range(len(conll_sentence))]
                    lstm_vecs.append(self.deep_lstms.transduce([concatenate([entry.vec, head_vec[j] if j==i else inputVector([0]*self.wdims), indicator[j]]) for j,entry in enumerate(conll_sentence)]))

                exprs = self.__evaluate(lstm_vecs)
                scores = np.array([[output.scalar_value() for output in exprsRow] for exprsRow in exprs])
                heads = decoder.parse_proj(scores)

                for entry, head in zip(conll_sentence, heads):
                    entry.pred_parent_id = head
                    entry.pred_relation = '_'

                dump = False
                for modifier, head in enumerate(heads[1:]):
                    exprs = self.__evaluateLabel(lstm_vecs, head, modifier+1)
                    scores = exprs.value()
                    conll_sentence[modifier+1].pred_relation = self.irels[max(enumerate(scores), key=itemgetter(1))[0]]

                renew_cg()
                if not dump:
                    yield sentence

    def Train(self, conll_path):
        errors = 0
        batch = 0
        eloss = 0.0
        mloss = 0.0
        eerrors = 0
        etotal = 0
        start = time.time()

        with open(conll_path, 'r') as conllFP:
            shuffledData = list(read_conll(conllFP))
            random.shuffle(shuffledData)

            errs = []
            lerrs = []
            eeloss = 0.0

            for iSentence, sentence in enumerate(shuffledData):
                if iSentence % 100 == 0 and iSentence != 0:
                    print 'Processing sentence number:', iSentence, 'Loss:', eloss / etotal, 'Errors:', (float(eerrors)) / etotal, 'Time', time.time()-start
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0
                    lerrors = 0
                    ltotal = 0

                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                head_vec = []
                for entry in conll_sentence:
                    c = float(self.wordsCount.get(entry.norm, 0))
                    dropFlag = (random.random() < (c/(0.25+c)))
                    wordvec = self.wlookup[int(self.vocab.get(entry.norm, 0)) if dropFlag else 0] if self.wdims > 0 else None
                    head_vec.append(self.hlookup[int(self.vocab.get(entry.norm, 0)) if dropFlag else 0] if self.wdims > 0 else None)
                    posvec = self.plookup[int(self.pos[entry.pos])] if self.pdims > 0 else None
                    evec = None

                    if self.external_embedding is not None:
                        evec = self.elookup[self.extrnd.get(entry.form, self.extrnd.get(entry.norm, 0)) if (dropFlag or (random.random() < 0.5)) else 0]
                    entry.vec = concatenate(filter(None, [wordvec, posvec, evec]))
                    if self.dropout:
                        entry.vec = dropout(entry.vec, self.dropout_prob)

                if self.dropout:
                    self.deep_lstms.set_dropout(self.dropout_prob)
                lstm_vecs = list()
                for i in range(len(conll_sentence)):
                    indicator = [scalarInput(1) if j == i else scalarInput(0) for j in range(len(conll_sentence))]
                    lstm = self.deep_lstms.transduce([concatenate([entry.vec, head_vec[j] if j==i else inputVector([0]*self.wdims), indicator[j]]) for j, entry in enumerate(conll_sentence)])
                    lstm_vecs.append(lstm)
                exprs = self.__evaluate(lstm_vecs)
                gold = [entry.parent_id for entry in conll_sentence]

                rexprs_list = []
                for modifier, head in enumerate(gold[1:]):
                    rexprs = self.__evaluateLabel(lstm_vecs, head, modifier+1)
                    rexprs_list.append(rexprs)

                for modifier, head in enumerate(gold[1:]):
                    rscores = rexprs_list[modifier].value()
                    goldLabelInd = self.rels[conll_sentence[modifier+1].relation]
                    wrongLabelInd = max(((l, scr) for l, scr in enumerate(rscores) if l != goldLabelInd), key=itemgetter(1))[0]
                    if rscores[goldLabelInd] < rscores[wrongLabelInd] + 1:
                        lerrs.append(rexprs[wrongLabelInd] - rexprs[goldLabelInd])

                scores = np.array([[output.scalar_value() for output in exprsRow] for exprsRow in exprs])
                heads = decoder.parse_proj(scores, gold if self.costaugFlag else None)
                e = sum([1 for h, g in zip(heads[1:], gold[1:]) if h != g])
                eerrors += e
                if e > 0:
                    loss = [(exprs[h][i] - exprs[g][i]) for i, (h,g) in enumerate(zip(heads, gold)) if h != g] # * (1.0/float(e))
                    eloss += (e)
                    mloss += (e)
                    errs.extend(loss)

                etotal += len(conll_sentence)

                if iSentence % 1 == 0 or len(errs) > 0 or len(lerrs) > 0:
                    eeloss = 0.0

                    if len(errs) > 0 or len(lerrs) > 0:
                        eerrs = (esum(errs + lerrs)) #* (1.0/(float(len(errs))))
                        eerrs.scalar_value()
                        eerrs.backward()
                        self.trainer.update()
                        errs = []
                        lerrs = []

                    renew_cg()

        if len(errs) > 0:
            eerrs = (esum(errs + lerrs)) #* (1.0/(float(len(errs))))
            eerrs.scalar_value()
            eerrs.backward()
            self.trainer.update()

            errs = []
            lerrs = []
            eeloss = 0.0

            renew_cg()

        self.trainer.update_epoch()
        print "Loss: ", mloss/iSentence
