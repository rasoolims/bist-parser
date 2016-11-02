from optparse import OptionParser
from arc_hybrid import ArcHybridLSTM
import pickle, utils, os, time, sys
from utils import ParseForest, read_conll, write_conll

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE",
                      default="../data/PTB_SD_3_3_0/train.conll")
    parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE",
                      default="")
    parser.add_option("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE",
                      default="../data/PTB_SD_3_3_0/test.conll")
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="barchybrid.model")
    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=100)
    parser.add_option("--pembedding", type="int", dest="pembedding_dims", default=25)
    parser.add_option("--rembedding", type="int", dest="rembedding_dims", default=25)
    parser.add_option("--action_dim", type="int", dest="action_dim", default=25)
    parser.add_option("--lang_embedding", type="int", dest="lang_embedding_dims", default=16)
    parser.add_option("--epochs", type="int", dest="epochs", default=30)
    parser.add_option("--hidden", type="int", dest="hidden_units", default=100)
    parser.add_option("--hidden2", type="int", dest="hidden2_units", default=0)
    parser.add_option("--k", type="int", dest="window", default=3)
    parser.add_option("--lr", type="float", dest="learning_rate", default=0.1)
    parser.add_option("--outdir", type="string", dest="output", default="results")
    parser.add_option("--activation", type="string", dest="activation", default="tanh")
    parser.add_option("--lstmlayers", type="int", dest="lstm_layers", default=2)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=200)
    parser.add_option("--dynet-seed", type="int", dest="seed", default=7)
    parser.add_option("--disableoracle", action="store_false", dest="oracle", default=True)
    parser.add_option("--disableblstm", action="store_false", dest="blstmFlag", default=True)
    parser.add_option("--bibi-lstm", action="store_true", dest="bibiFlag", default=False)
    parser.add_option("--usehead", action="store_true", dest="headFlag", default=False)
    parser.add_option("--userlmost", action="store_true", dest="rlFlag", default=False)
    parser.add_option("--userl", action="store_true", dest="rlMostFlag", default=False)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--partial", action="store_true", dest="partialFlag", default=False)
    parser.add_option("--dynet_mem", type="int", dest="dynet_mem", default=512)
    parser.add_option("--drop-out", type="float", dest="dropout", default=0)

    (options, args) = parser.parse_args()
    print 'Using external embedding:', options.external_embedding

    shuffledData = list(read_conll(conllFP, True))
    random.shuffle(shuffledData)

    print 'Preparing vocab'
    words, w2i, pos, rels, langs = utils.vocab(options.conll_train)

    with open(os.path.join(options.output, options.params), 'w') as paramsfp:
        pickle.dump((words, w2i, pos, rels, langs, options), paramsfp)
    print 'Finished collecting vocab'

    print 'Initializing blstm arc hybrid:'
    parser = ArcHybridLSTM(words, pos, rels, langs, w2i, options)

    for epoch in xrange(options.epochs):
        print 'Starting epoch for confidence', epoch
        parser.TrainConfidence(options.conll_train)
        parser.Save(os.path.join(options.output, options.model + str(epoch + 1)))
