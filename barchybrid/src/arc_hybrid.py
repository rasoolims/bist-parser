from dynet import *
from utils import ParseForest, read_conll, write_conll
from operator import itemgetter
from itertools import chain
import utils, time, random
import numpy as np

class ArcHybridLSTM:
    SHIFT = 2
    LEFT_ARC = 0
    RIGHT_ARC = 1

    def __init__(self, words, pos, rels, langs, w2i, options):
        self.model = Model()
        self.trainer = AdamTrainer(self.model)
        random.seed(1)

        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify,
                            'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}
        self.activation = self.activations[options.activation]
        self.dropout_prob = options.dropout
        print 'dropout prob', self.dropout_prob

        self.ldims = options.lstm_dims * 2
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.rdims = options.rembedding_dims
        self.langdims = options.lang_embedding_dims
        self.layers = options.lstm_layers
        self.wordsCount = words
        self.vocab = {word: ind + 3 for word, ind in w2i.iteritems()}
        self.pos = {word: ind + 3 for ind, word in enumerate(pos)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.langs = {word: ind for ind, word in enumerate(langs)}
        self.irels = rels
        self.actionDim = options.action_dim

        self.headFlag = options.headFlag

        self.nnvecs = (1 if self.headFlag else 0) + 2

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
            self.extrn_lookup = self.model.add_lookup_parameters((len(self.external_embedding) + 3, self.edim))
            for word, i in self.extrnd.iteritems():
                self.extrn_lookup.init_row(i, self.external_embedding[word])
            self.extrnd['*PAD*'] = 1
            self.extrnd['*INITIAL*'] = 2

            print 'Load external embedding. Vector dimensions', self.edim

        dims = self.wdims + self.pdims + (self.edim if self.external_embedding is not None else 0)

        self.surfaceBuilders = [LSTMBuilder(1, dims, self.ldims * 0.5, self.model),
                                LSTMBuilder(1, dims, self.ldims * 0.5, self.model)]
        self.bsurfaceBuilders = [LSTMBuilder(1, self.ldims, self.ldims * 0.5, self.model),
                                 LSTMBuilder(1, self.ldims, self.ldims * 0.5, self.model)]

        self.transition_lstm = LSTMBuilder(1, (3*self.ldims) + self.actionDim, self.ldims * 0.5, self.model)

        self.vocab['*PAD*'] = 1
        self.pos['*PAD*'] = 1

        self.vocab['*INITIAL*'] = 2
        self.pos['*INITIAL*'] = 2

        self.word_lookup = self.model.add_lookup_parameters((len(words) + 3, self.wdims))
        self.pos_lookup = self.model.add_lookup_parameters((len(pos) + 3, self.pdims))
        self.rels_lookup = self.model.add_lookup_parameters((len(rels), self.rdims))
        self.lang_lookup = self.model.add_lookup_parameters((len(langs), self.langdims))
        self.action_lookup = self.model.add_lookup_parameters((2 * (len(self.irels) + 0) + 1, self.actionDim))

        self.word2lstm_ = self.model.add_parameters((self.ldims, self.wdims + self.pdims + (self.edim if self.external_embedding is not None else 0)))
        self.word2half_lstm_ = self.model.add_parameters((self.ldims, self.wdims + self.pdims + (self.edim if self.external_embedding is not None else 0)))
        self.word2lstmbias_ = self.model.add_parameters((self.ldims))
        self.word2halflstmbias_ = self.model.add_parameters((self.ldims))
        self.lstm2lstm_ = self.model.add_parameters((self.ldims, self.ldims * self.nnvecs + self.rdims))
        self.lstm2lstmbias_ = self.model.add_parameters((self.ldims))

        self.trans_outLayer_ = self.model.add_parameters((2, self.langdims + self.ldims*.5))
        self.trans_outBias_ = self.model.add_parameters((2))

    def __evaluate_transitions(self, stack, buf, langVector):
        trans_vec = self.transition_lstm.initial_state()

        while len(buf) > 0 or len(stack) > 1:
            s1 = [stack.roots[-2]] if len(stack) > 1 else []
            s0 = [stack.roots[-1]] if len(stack) > 0 else []
            b = [buf.roots[0]] if len(buf) > 0 else []
            beta = buf.roots[1:] if len(buf) > 1 else []
            can_left = True if len(stack) > 0 and len(buf) > 0 else False
            can_right = True if len(stack) > 1  else False

            left_cost = (len([h for h in s1 + beta if h.id == s0[0].parent_id]) +
                         len([d for d in b + beta if d.parent_id == s0[0].id])) if can_left else 1
            right_cost = (len([h for h in b + beta if h.id == s0[0].parent_id]) +
                          len([d for d in b + beta if d.parent_id == s0[0].id])) if can_right else 1

            topStack = [stack.roots[-i - 1].raw_lstms if len(stack) > i else [self.half_empty] for i in xrange(2)]
            topBuffer = [buf.roots[i].raw_lstms if len(buf) > i else [self.half_empty] for i in xrange(1)]

            selected_action = 1 if left_cost==0 else 2 if right_cost==0 else 0
            label = '' if selected_action==0 else buf.roots[0].relation if selected_action == 1 else stack.roots[-1].relation
            gold_action = 0 if selected_action == 0 else (selected_action-1)*len(self.rels)+self.rels[label]
            action_vec = concatenate([lookup(self.action_lookup, gold_action)])
            input = concatenate([action_vec, concatenate(list(chain(*(topStack + topBuffer))))])
            trans_vec = trans_vec.add_input(input)

            if selected_action == 0:
                stack.roots.append(buf.roots[0])
                del buf.roots[0]

            elif selected_action == 1:
                child = stack.roots.pop()
                parent = buf.roots[0]
                child.pred_parent_id = parent.id
                child.pred_relation = label

            elif selected_action == 2:
                child = stack.roots.pop()
                parent = stack.roots[-1]
                child.pred_parent_id = parent.id
                child.pred_relation = label

        return softmax(self.trans_outLayer * concatenate([langVector, trans_vec.output()]) + self.trans_outBias)

    def Save(self, filename):
        self.model.save(filename)

    def Load(self, filename):
        self.model.load(filename)

    def Init(self):
        self.word2lstm = parameter(self.word2lstm_)
        self.word2half_lstm = parameter(self.word2half_lstm_)
        self.lstm2lstm = parameter(self.lstm2lstm_)

        self.word2lstmbias = parameter(self.word2lstmbias_)
        self.word2halflstmbias = parameter(self.word2halflstmbias_)
        self.lstm2lstmbias = parameter(self.lstm2lstmbias_)

        self.trans_outLayer = parameter(self.trans_outLayer_)
        self.trans_outBias = parameter(self.trans_outBias_)

        evec = lookup(self.extrn_lookup, 1) if self.external_embedding is not None else None
        paddingWordVec = lookup(self.word_lookup, 1)
        paddingPosVec = lookup(self.pos_lookup, 1) if self.pdims > 0 else None

        paddingVec = tanh(
            self.word2lstm * concatenate(filter(None, [paddingWordVec, paddingPosVec, evec])) + self.word2lstmbias)
        self.empty = paddingVec if self.nnvecs == 1 else concatenate([paddingVec for _ in xrange(self.nnvecs)])
        self.half_empty = tanh(
            self.word2half_lstm * concatenate(filter(None, [paddingWordVec, paddingPosVec, evec])) + self.word2halflstmbias)

    def getWordEmbeddings(self, sentence, train):
        for root in sentence:
            c = float(self.wordsCount.get(root.norm, 0))
            dropFlag = not train or (random.random() < (c / (0.25 + c)))
            root.wordvec = lookup(self.word_lookup, int(self.vocab.get(root.norm, 0)) if dropFlag else 0)
            root.posvec = lookup(self.pos_lookup, int(self.pos[root.pos])) if self.pdims > 0 else None

            if self.external_embedding is not None:
                if not dropFlag and random.random() < 0.5:
                    root.evec = lookup(self.extrn_lookup, 0)
                elif root.form in self.external_embedding:
                    root.evec = lookup(self.extrn_lookup, self.extrnd[root.form], update=True)
                elif root.norm in self.external_embedding:
                    root.evec = lookup(self.extrn_lookup, self.extrnd[root.norm], update=True)
                else:
                    root.evec = lookup(self.extrn_lookup, 0)
            else:
                root.evec = None
            root.ivec = concatenate(filter(None, [root.wordvec, root.posvec, root.evec]))

        forward = self.surfaceBuilders[0].initial_state()
        backward = self.surfaceBuilders[1].initial_state()

        for froot, rroot in zip(sentence, reversed(sentence)):
            forward = forward.add_input(froot.ivec)
            backward = backward.add_input(rroot.ivec)
            froot.fvec = forward.output()
            rroot.bvec = backward.output()
        for root in sentence:
            root.vec = concatenate([root.fvec, root.bvec])

        bforward = self.bsurfaceBuilders[0].initial_state()
        bbackward = self.bsurfaceBuilders[1].initial_state()

        for froot, rroot in zip(sentence, reversed(sentence)):
            bforward = bforward.add_input(froot.vec)
            bbackward = bbackward.add_input(rroot.vec)
            froot.bfvec = bforward.output()
            rroot.bbvec = bbackward.output()
        for root in sentence:
            root.vec = concatenate([root.bfvec, root.bbvec])
       
        langVec = concatenate(filter(None, [lookup(self.lang_lookup, int(self.langs[sentence[0].lang_id])) if self.langdims > 0 else None]))
        return langVec

    def TrainConfidence(self, conll_path):
        mloss = 0.0
        eloss = 0.0
        etotal = 0
        start = time.time()
        distance = 0
        with open(conll_path, 'r') as conllFP:
            shuffledData = list(read_conll(conllFP, True))
            random.shuffle(shuffledData)
            errs = []
            self.Init()

            for iSentence, sentence in enumerate(shuffledData):
                if iSentence % 100 == 0 and iSentence != 0:
                    print 'Processing sentence number:', iSentence, 'Loss:', eloss / etotal, 'Distance:',distance/etotal, 'Time', time.time() - start
                    start = time.time()
                    eloss = 0.0
                    etotal = 0
                    distance = 0

                sentence = sentence[1:] + [sentence[0]]
                langVector = self.getWordEmbeddings(sentence, True)
                stack = ParseForest([])
                buf = ParseForest(sentence)
                for root in sentence:
                    root.lstms = [root.vec for _ in xrange(self.nnvecs)]
                    root.raw_lstms = [root.vec]

                eps = scalarInput(1e-20)
                result = self.__evaluate_transitions(stack, buf, langVector)
                extrinsic_weight = scalarInput(sentence[0].weight)+eps
                other_weight = scalarInput(1.0 - sentence[0].weight)+eps
                loss = extrinsic_weight * (log(extrinsic_weight)-log(result[0]+eps)) + other_weight * (log(other_weight)-log(result[1]+eps))
                eloss += loss.value()
                mloss += loss.value()
                errs.append(loss)
                distance+= abs(sentence[0].weight - result[0].value())
                etotal+=1

                if len(errs) > 50:  # or True:
                    # eerrs = ((esum(errs)) * (1.0/(float(len(errs)))))
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

            errs = []
            etotal+= 1
            renew_cg()

        self.trainer.update_epoch()
        print "Loss: ", mloss / iSentence

    def Predict(self, conll_path):
        with open(conll_path, 'r') as conllFP:
            data = list(read_conll(conllFP, True))

            for iSentence, sentence in enumerate(data):
                self.Init()
                sentence = sentence[1:] + [sentence[0]]
                langVector = self.getWordEmbeddings(sentence, True)
                stack = ParseForest([])
                buf = ParseForest(sentence)
                for root in sentence:
                    root.lstms = [root.vec for _ in xrange(self.nnvecs)]
                    root.raw_lstms = [root.vec]

                result = self.__evaluate_transitions(stack, buf, langVector)

                for s in sentence:
                    s.weight = result[0].value()
                renew_cg()
                yield [sentence[-1]] + sentence[:-1]
