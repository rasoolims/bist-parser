from optparse import OptionParser
import pickle, utils, mstlstm, os, os.path, time


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE", default="../data/en-universal-train.conll.ptb")
    parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE", default="../data/en-universal-dev.conll.ptb")
    parser.add_option("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE", default="../data/en-universal-test.conll.ptb")
    parser.add_option("--output", dest="conll_output",  metavar="FILE", default=None)
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="parser.model")
    parser.add_option("--we", type="int", dest="we", default=100)
    parser.add_option("--pe", type="int", dest="pe", default=25)
    parser.add_option("--re", type="int", dest="re", default=25)
    parser.add_option("--epochs", type="int", dest="epochs", default=30)
    parser.add_option("--hidden", type="int", dest="hidden_units", default=100)
    parser.add_option("--hidden2", type="int", dest="hidden2_units", default=0)
    parser.add_option("--lr", type="float", dest="lr", default=0.001)
    parser.add_option("--beta1", type="float", dest="beta1", default=0.9)
    parser.add_option("--beta2", type="float", dest="beta2", default=0.999)
    parser.add_option("--dropout", type="float", dest="dropout", default=0.0)
    parser.add_option("--outdir", type="string", dest="output", default="results")
    parser.add_option("--activation", type="string", dest="activation", default="tanh")
    parser.add_option("--layer", type="int", dest="layer", default=2)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=125)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--disablecostaug", action="store_false", dest="costaugFlag", default=True)
    parser.add_option("--dynet-seed", type="int", dest="seed", default=0)
    parser.add_option("--dynet-mem", type="int", dest="mem", default=0)
    parser.add_option("--dynet-autobatch", type="int", dest="dynet-autobatch", default=0)

    (options, args) = parser.parse_args()

    print 'Using external embedding:', options.external_embedding
    if options.predictFlag:
        with open(options.params, 'r') as paramsfp:
            words, w2i, pos, rels, stored_opt = pickle.load(paramsfp)
        stored_opt.external_embedding = options.external_embedding
        print 'Initializing lstm mstparser:'
        parser = mstlstm.MSTParserLSTM(words, pos, rels, w2i, stored_opt)
        parser.Load(options.model)
        ts = time.time()
        test_res = list(parser.Predict(options.conll_test))
        te = time.time()
        print 'Finished predicting test.', te-ts, 'seconds.'
        utils.write_conll(options.conll_output, test_res)
    else:
        print 'Preparing vocab'
        words, w2i, pos, rels = utils.vocab(options.conll_train)

        with open(os.path.join(options.output, options.params), 'w') as paramsfp:
            pickle.dump((words, w2i, pos, rels, options), paramsfp)
        print 'Finished collecting vocab'

        print 'Initializing lstm mstparser:'
        parser = mstlstm.MSTParserLSTM(words, pos, rels, w2i, options)
        best_acc = -float('inf')
        for epoch in xrange(options.epochs):
            print 'Starting epoch', epoch
            parser.Train(options.conll_train)
            devpath = os.path.join(options.output, 'dev_epoch_out')
            utils.write_conll(devpath, parser.Predict(options.conll_dev))
            acc = utils.eval(options.conll_dev, devpath)
            print 'currect UAS', acc
            if acc > best_acc:
                print 'saving model', acc
                best_acc = acc
                parser.Save(os.path.join(options.output, os.path.basename(options.model)))
