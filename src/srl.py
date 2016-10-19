from dynet import *
from utils import read_conll
from operator import itemgetter
from itertools import chain
import utils, time, random
import numpy as np

class SRLLSTM:
    def __init__(self, words, lemmas, pos, depRels, rels, w2i, l2i, options):
        self.model = Model()
        self.trainer = AdamTrainer(self.model)
        random.seed(1)

        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify,
                            'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}
        self.activation = self.activations[options.activation]

        self.oracle = options.oracle
        self.ldims = options.lstm_dims * 2
        self.wdims = options.wembedding_dims
        self.lemDims = options.lem_embedding_dims
        self.pdims = options.pembedding_dims
        self.deprdims = options.deprembedding_dims
        self.rdims = options.rembedding_dims
        self.layers = options.lstm_layers
        self.wordsCount = words
        self.vocab = {word: ind + 3 for word, ind in w2i.iteritems()}
        self.lemmas = {lemma: ind + 3 for lemma, ind in l2i.iteritems()}
        self.pos = {word: ind + 3 for ind, word in enumerate(pos)}
        self.deprels = {word: ind for ind, word in enumerate(depRels)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels

        self.headFlag = options.headFlag
        self.rlMostFlag = options.rlMostFlag
        self.rlFlag = options.rlFlag
        self.k = 4
        self.nnvecs = 2

        self.external_embedding = None
        if options.external_embedding is not None:
            external_embedding_fp = open(options.external_embedding, 'r')
            external_embedding_fp.readline()
            self.external_embedding = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in
                                       external_embedding_fp}
            external_embedding_fp.close()

            self.edim = len(self.external_embedding.values()[0])
            self.noextrn = [0.0 for _ in xrange(self.edim)]
            self.extrnd = {word: i + 3 for i, word in enumerate(self.external_embedding)}
            self.extrn = self.model.add_lookup_parameters((len(self.external_embedding) + 3, self.edim))
            for word, i in self.extrnd.iteritems():
                self.extrn.init_row(i, self.external_embedding[word])
            self.extrnd['*PAD*'] = 1
            self.extrnd['*INITIAL*'] = 2

            print 'Load external embedding. Vector dimensions', self.edim

        dims = self.wdims + self.lemDims + self.pdims + (self.edim if self.external_embedding is not None else 0)
        self.blstmFlag = options.blstmFlag
        self.bibiFlag = options.bibiFlag

        if self.bibiFlag:
            self.surfaceBuilders = [LSTMBuilder(1, dims, self.ldims * 0.5, self.model),
                                    LSTMBuilder(1, dims, self.ldims * 0.5, self.model)]
            self.bsurfaceBuilders = [LSTMBuilder(1, self.ldims, self.ldims * 0.5, self.model),
                                     LSTMBuilder(1, self.ldims, self.ldims * 0.5, self.model)]
        elif self.blstmFlag:
            if self.layers > 0:
                self.surfaceBuilders = [LSTMBuilder(self.layers, dims, self.ldims * 0.5, self.model),
                                        LSTMBuilder(self.layers, dims, self.ldims * 0.5, self.model)]
            else:
                self.surfaceBuilders = [SimpleRNNBuilder(1, dims, self.ldims * 0.5, self.model),
                                        LSTMBuilder(1, dims, self.ldims * 0.5, self.model)]

        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units
        self.vocab['*PAD*'] = 1
        self.pos['*PAD*'] = 1
        self.lemmas['*PAD*'] = 1
        self.deprels['*PAD*'] = 1

        self.vocab['*INITIAL*'] = 2
        self.pos['*INITIAL*'] = 2
        self.lemmas['*INITIAL*'] = 2
        self.deprels['*INITIAL*'] = 2

        self.wordEmbeddings = self.model.add_lookup_parameters((len(words) + 3, self.wdims))
        self.lemmaEmbeddings = self.model.add_lookup_parameters((len(lemmas) + 3, self.lemDims))
        self.posEmbedding = self.model.add_lookup_parameters((len(pos) + 3, self.pdims))
        self.depRelEmbedding = self.model.add_lookup_parameters((len(depRels), self.deprdims))
        self.semRelEmbedding = self.model.add_lookup_parameters((len(rels), self.rdims))

        self.word2lstm_ = self.model.add_parameters((self.ldims, self.wdims + self.lemDims + self.pdims + (self.edim if self.external_embedding is not None else 0)))
        self.word2lstmbias_ = self.model.add_parameters((self.ldims))
        self.lstm2lstm_ = self.model.add_parameters((self.ldims, self.ldims * self.nnvecs + self.rdims))
        self.lstm2lstmbias_ = self.model.add_parameters((self.ldims))

        self.hidLayer_ = self.model.add_parameters((self.hidden_units, self.ldims * self.nnvecs * self.k))
        self.hidBias_ = self.model.add_parameters((self.hidden_units))

        self.hid2Layer_ = self.model.add_parameters((self.hidden2_units, self.hidden_units))
        self.hid2Bias_ = self.model.add_parameters((self.hidden2_units))

        self.outLayer_ = self.model.add_parameters((2, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))
        self.outBias_ = self.model.add_parameters((2))

        self.rhidLayer_ = self.model.add_parameters((self.hidden_units, self.ldims * self.nnvecs * self.k))
        self.rhidBias_  = self.model.add_parameters((self.hidden_units))

        self.rhid2Layer_ = self.model.add_parameters((self.hidden2_units, self.hidden_units))
        self.rhid2Bias_ = self.model.add_parameters((self.hidden2_units))

        self.routLayer_ =  self.model.add_parameters((2 * (len(self.irels) + 0) + 1, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))
        self.routBias_ = self.model.add_parameters((2 * (len(self.irels) + 0) + 1))

        self.word2lstm = parameter(self.word2lstm_)
        self.word2lstmbias = parameter(self.word2lstmbias_)
        self.lstm2lstm =  parameter(self.lstm2lstm_)
        self.lstm2lstmbias =  parameter(self.lstm2lstmbias_)
        self.hidLayer = parameter(self.hidLayer_)
        self.hidBias = parameter(self.hidBias_)
        self.hid2Layer = parameter(self.hid2Layer_)
        self.hid2Bias = parameter(self.hid2Bias_)
        self.outLayer = parameter(self.outLayer_)
        self.outBias = parameter(self.outBias_)
        self.rhidLayer = parameter(self.rhidLayer_)
        self.rhidBias = parameter(self.rhidBias_)
        self.rhid2Layer = parameter(self.rhid2Layer_)
        self.rhid2Bias = parameter(self.rhid2Bias_)
        self.routLayer = parameter(self.routLayer_)
        self.routBias = parameter(self.routBias_)

    def __evaluate(self, sentence, pred_index, arg_index):
        pred_vec = [sentence.entries[pred_index].lstms]
        arg_vec = [sentence.entries[arg_index].lstms]
        pred_head = sentence.head(pred_index)
        pred_head_vec = [sentence.entries[pred_head].lstms if pred_head>=0 else [self.empty]]
        arg_head = sentence.head(arg_index)
        arg_head_vec = [sentence.entries[arg_head].lstms if arg_head >= 0 else [self.empty]]

        input = concatenate(list(chain(*(pred_vec + arg_vec + pred_head_vec + arg_head_vec))))

        if self.hidden2_units > 0:
            routput = (self.routLayer * self.activation(self.rhid2Bias + self.rhid2Layer * self.activation(
                self.rhidLayer * input + self.rhidBias)) + self.routBias)
        else:
            routput = (self.routLayer * self.activation(self.rhidLayer * input + self.rhidBias) + self.routBias)

        if self.hidden2_units > 0:
            output = (self.outLayer * self.activation(
                self.hid2Bias + self.hid2Layer * self.activation(self.hidLayer * input + self.hidBias)) + self.outBias)
        else:
            output = (self.outLayer * self.activation(self.hidLayer * input + self.hidBias) + self.outBias)

        scrs, uscrs = routput.value(), output.value()

        uscrs0 = uscrs[0]
        uscrs1 = uscrs[1]
        output0 = output[0]
        output1 = output[1]
        return  [[(rel, 0, scrs[1 + j * 2] + uscrs1, routput[1 + j * 2] + output1) for j, rel in enumerate(self.irels)],
                   [('_', 1, scrs[0] + uscrs0, routput[0] + output0)]]

    def Save(self, filename):
        self.model.save(filename)

    def Load(self, filename):
        self.model.load(filename)

    def Init(self):
        self.word2lstm = parameter(self.word2lstm_)
        self.word2lstmbias = parameter(self.word2lstmbias_)
        self.lstm2lstm = parameter(self.lstm2lstm_)
        self.lstm2lstmbias = parameter(self.lstm2lstmbias_)
        self.hidLayer = parameter(self.hidLayer_)
        self.hidBias = parameter(self.hidBias_)
        self.hid2Layer = parameter(self.hid2Layer_)
        self.hid2Bias = parameter(self.hid2Bias_)
        self.outLayer = parameter(self.outLayer_)
        self.outBias = parameter(self.outBias_)
        self.rhidLayer = parameter(self.rhidLayer_)
        self.rhidBias = parameter(self.rhidBias_)
        self.rhid2Layer = parameter(self.rhid2Layer_)
        self.rhid2Bias = parameter(self.rhid2Bias_)
        self.routLayer = parameter(self.routLayer_)
        self.routBias = parameter(self.routBias_)

        evec = lookup(self.extrn, 1) if self.external_embedding is not None else None
        paddingWordVec = lookup(self.wordEmbeddings, 1)
        paddingLemmaVec = lookup(self.lemmaEmbeddings, 1)
        paddingPosVec = lookup(self.posEmbedding, 1)
        paddingVec = tanh(
            self.word2lstm * concatenate(filter(None, [paddingWordVec, paddingLemmaVec, paddingPosVec, evec])) + self.word2lstmbias)
        self.empty = paddingVec if self.nnvecs == 1 else concatenate([paddingVec for _ in xrange(self.nnvecs)])

    def getWordEmbeddings(self, sentence, train):
        for root in sentence:
            c = float(self.wordsCount.get(root.norm, 0))
            dropFlag = not train or (random.random() < (c / (0.25 + c)))
            root.wordvec = lookup(self.wordEmbeddings, int(self.vocab.get(root.norm, 0)) if dropFlag else 0)
            root.lemmaVec = lookup(self.lemmaEmbeddings, int(self.lemmas.get(root.lemmaNorm, 0)) if dropFlag else 0)
            root.posvec = lookup(self.posEmbedding, int(self.pos[root.pos])) if self.pdims > 0 else None

            if self.external_embedding is not None:
                if not dropFlag and random.random() < 0.5:
                    root.evec = lookup(self.extrn, 0)
                elif root.form in self.external_embedding:
                    root.evec = lookup(self.extrn, self.extrnd[root.form], update=True)
                elif root.norm in self.external_embedding:
                    root.evec = lookup(self.extrn, self.extrnd[root.norm], update=True)
                else:
                    root.evec = lookup(self.extrn, 0)
            else:
                root.evec = None
            root.ivec = concatenate(filter(None, [root.wordvec, root.lemmaVec, root.posvec, root.evec]))

        if self.blstmFlag:
            forward = self.surfaceBuilders[0].initial_state()
            backward = self.surfaceBuilders[1].initial_state()

            for froot, rroot in zip(sentence, reversed(sentence)):
                forward = forward.add_input(froot.ivec)
                backward = backward.add_input(rroot.ivec)
                froot.fvec = forward.output()
                rroot.bvec = backward.output()
            for root in sentence:
                root.vec = concatenate([root.fvec, root.bvec])

            if self.bibiFlag:
                bforward = self.bsurfaceBuilders[0].initial_state()
                bbackward = self.bsurfaceBuilders[1].initial_state()

                for froot, rroot in zip(sentence, reversed(sentence)):
                    bforward = bforward.add_input(froot.vec)
                    bbackward = bbackward.add_input(rroot.vec)
                    froot.bfvec = bforward.output()
                    rroot.bbvec = bbackward.output()
                for root in sentence:
                    root.vec = concatenate([root.bfvec, root.bbvec])
        else:
            for root in sentence:
                root.ivec = (self.word2lstm * root.ivec) + self.word2lstmbias
                root.vec = tanh(root.ivec)

    def Predict(self, conll_path):
        for iSentence, sentence in enumerate(read_conll(conll_path)):
            self.Init()
            self.getWordEmbeddings(sentence.entries, False)
            for root in sentence.entries:
                root.lstms = [root.vec for _ in xrange(self.nnvecs)]
            for p in range(len(sentence.predicates)):
                predicate = sentence.predicates[p]
                for arg in range(1, len(sentence.entries)):
                    scores = self.__evaluate(sentence, predicate, arg)
                    sentence.entries[arg].predicateList[p] = max(chain(*scores), key=itemgetter(2))[0]
            renew_cg()
            yield  sentence

    def Train(self, conll_path):
        mloss = 0.0
        eloss = 0.0
        eerrors = 0
        lerrors = 0
        etotal = 0
        ninf = -float('inf')
        start = time.time()
        shuffledData = list(read_conll(conll_path))
        random.shuffle(shuffledData)
        errs = []
        self.Init()
        for iSentence, sentence in enumerate(shuffledData):
            if iSentence % 100 == 0:
                try:
                    print 'Processing sentence number:', iSentence, 'Loss:', eloss / etotal, 'Errors:', (float(
                        eerrors)) / etotal, 'Labeled Errors:', (float(lerrors) / etotal), 'Time', time.time() - start
                except:
                    print 'sentence', iSentence
                start = time.time()
                eerrors = 0
                eloss = 0.0
                etotal = 0
                lerrors = 0
            self.getWordEmbeddings(sentence.entries, True)
            for root in sentence.entries:
                root.lstms = [root.vec for _ in xrange(self.nnvecs)]
            for p in range(1, len(sentence.predicates)):
                predicate = sentence.predicates[p]
                for arg in range(1, len(sentence.entries)):
                    scores = self.__evaluate(sentence, predicate, arg)
                    best = max(chain(*scores), key=itemgetter(2))
                    gold = sentence.entries[arg].predicateList[p]
                    predicted = best[0]

                    if gold != predicted:
                        gold_score = 0
                        g_s = 0
                        if gold == '_':
                            gold_score = scores[1][0][3]
                            g_s = scores[1][0][2]
                        else:
                            for item in scores[0]:
                                if item[0]==gold:
                                    gold_score = item[3]
                                    g_s = item[2]
                                    break

                        if gold != '_' and predicted!='_':
                            lerrors+=1
                        else:
                            lerrors += 1
                            eerrors += 1

                        loss = best[3] - gold_score
                        mloss += 1.0 + best[2] - g_s
                        eloss += 1.0 + best[2] - g_s
                        errs.append(loss)
                    etotal+= 1
            if len(errs) > 50:
                eerrs = esum(errs)
                scalar_loss = eerrs.scalar_value()
                eerrs.backward()
                self.trainer.update()
                errs = []
                renew_cg()
                self.Init()

        if len(errs) > 0:
            eerrs = (esum(errs))  # * (1.0/(float(len(errs))))
            eerrs.scalar_value()
            eerrs.backward()
            self.trainer.update()
            renew_cg()

        self.trainer.update_epoch()
        print "Loss: ", mloss / iSentence
