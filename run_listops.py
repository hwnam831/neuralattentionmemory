import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import Options
import Models
from ListOpsDataset import GuidedListops
import STM
import NAM
from torch.utils.data import DataLoader
import time
import math
from transformer_generalization.layers.transformer.universal_transformer import UniversalTransformerEncoder
from transformer_generalization.layers.transformer.universal_relative_transformer import RelativeTransformerEncoderLayer

class LSTMListops(nn.Module):
    def __init__(self, model_size, vocab_size=16):
        super().__init__()
        assert model_size %2 == 0
        self.model_size = model_size
        self.embed = nn.Embedding(vocab_size, self.model_size)
        self.encoder = nn.LSTM(self.model_size, self.model_size//2, 1, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.attn = Models.AttentionLayer(model_size, model_size, model_size)
        self.decoder = nn.LSTM(self.model_size, self.model_size//2, 1, bidirectional=True)
        self.fc = nn.Linear(model_size, vocab_size*4)

    def forward(self, input):
        outputs = self.dropout(self.embed(input.permute(1,0)))
        outputs, state = self.encoder(outputs)
        outputs, _ = self.attn(outputs, outputs)
        outputs, state = self.decoder(self.dropout(outputs))
        return self.fc(self.dropout(outputs)).reshape(outputs.shape[0],outputs.shape[1],4,-1).permute(1,3,0,2)

class NAMTMListops(nn.Module):
    def __init__(self, dim, vocab_size, nhead=4, defalt_tapelen=32, option='default', debug=False, mem_size=64):
        super().__init__()
        self.dim=dim
        self.embedding = nn.Sequential(nn.Embedding(vocab_size, dim),
                                        nn.Linear(dim, dim),
                                     nn.Dropout(0.2))

        self.encnorm = nn.LayerNorm(self.dim)
        self.tm = NAM.NAMTuring(dim, n_tapes=nhead, default_tapelen=defalt_tapelen, mem_size=mem_size)
        self.fc = nn.Sequential(nn.LayerNorm(dim),
            nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(dim, 4*vocab_size))
        self.vocab_size = vocab_size

    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input):
        input2 = input.permute(1,0)

        src = self.embedding(input2)

        src = self.encnorm(src)
        outputs, tape = self.tm(src)
        #outputs, tape = self.tm2(src, tape_in=tape)

        #S,N,4,C to N,C,S,4
        return self.fc(outputs).reshape(outputs.shape[0],outputs.shape[1],4,-1).permute(1,3,0,2)

class NAMTMListops2(nn.Module):
    def __init__(self, dim, vocab_size, nhead=4, defalt_tapelen=32, option='default', debug=False, mem_size=64):
        super().__init__()
        self.dim=dim
        self.embedding = nn.Sequential(nn.Embedding(vocab_size, dim),
                                        nn.Linear(dim, dim),
                                     nn.Dropout(0.2))

        self.encnorm = nn.LayerNorm(self.dim)
        self.tm = NAM.NAMTuringNoJump(dim, n_tapes=nhead, default_tapelen=defalt_tapelen, mem_size=mem_size)
        self.fc = nn.Sequential(nn.LayerNorm(dim),
            nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(dim, 4*vocab_size))
        self.vocab_size = vocab_size

    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input):
        input2 = input.permute(1,0)

        src = self.embedding(input2)

        src = self.encnorm(src)
        outputs, tape = self.tm(src)
        #outputs, tape = self.tm2(src, tape_in=tape)

        #S,N,4,C to N,C,S,4
        return self.fc(outputs).reshape(outputs.shape[0],outputs.shape[1],4,-1).permute(1,3,0,2)

class UTListops(nn.Module):
    def __init__(self, model_size=64, nhead=4, num_layers=2, vocab_size=16):
        super().__init__()
        self.model_size=model_size
        self.vocab_size = vocab_size
        assert model_size%2 == 0
        self.embedding = nn.Embedding(vocab_size, model_size)
        self.cell = UniversalTransformerEncoder(RelativeTransformerEncoderLayer,
                                                depth=num_layers, d_model=model_size, nhead=nhead)
        self.fc = nn.Linear(model_size, vocab_size*4)

    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input):

        src = self.cell(self.embedding(input)) #UT takes N, S, C
        #N,S,4,C to N,C,S,4
        return self.fc(src).reshape(src.shape[0],src.shape[1],4,-1).permute(0,3,1,2)

def train(model, trainloader, criterion, optimizer, scheduler):
        model.train(mode=True)
        tcorrect = 0
        tlen     = 0
        tloss    = 0
        bits = 0.0

        for x,y in trainloader:
            xdata       = x.cuda()
            ydata       = y.cuda()
            optimizer.zero_grad()
            output      = model(xdata)

            loss        = criterion(output, ydata)
            loss.mean().backward()
            bits += (loss).sum().item()

            tloss       = tloss + loss.mean().item()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            
            pred        = output.argmax(axis=1)
            seqcorrect  = (pred==ydata).prod(-1).prod(-1)
            tcorrect    = tcorrect + seqcorrect.sum().item()
            tlen        = tlen + seqcorrect.nelement()
        scheduler.step()

        trainingResult = list()
        print('train seq acc:\t'+str(tcorrect/tlen))
        print('train loss:\t{}'.format(tloss/len(trainloader)))
        print('Current LR:' + str(scheduler.get_last_lr()[0]))
        trainingResult.append('train seq acc:\t'+str(tcorrect/tlen))
        trainingResult.append(str('train loss:\t{}'.format(tloss/len(trainloader))))
        trainingResult.append('Current LR:' + str(scheduler.get_last_lr()[0]))
        
        #Perplexity  = 2^bit
        print('Training Perplexity :\t{}'.format(math.exp((bits/tlen) * math.log(2)))) 
        trainingResult.append('Training Perplexity :\t{}'.format(math.exp((bits/tlen) * math.log(2))))
       


        return model, trainingResult


def validate(model, valloader, valloader2, testloader, args):
        vcorrect = 0
        vlen = 0
        vloss = 0
        vcorrect2 = 0
        vlen2 = 0
        vloss2 = 0
        model.train(mode=False)
        bits = 0.0

        bits2 = 0.0

        tcorrect = 0
        tlen = 0
        with torch.no_grad():
            for i,(x,y) in enumerate(valloader):
                xdata       = x.cuda()
                ydata2      = y.cuda()
                output      = model(xdata)
                # xdata <- masked index
                # ydata2 <- answer 

                loss        = F.cross_entropy(output, ydata2, reduction='none')
                vloss       = vloss + loss.mean().item()
                bits += (loss).sum().item()

                pred2       = output.argmax(axis=1)
                seqcorrect  = (pred2==ydata2).prod(-1).prod(-1)
                vcorrect = vcorrect + seqcorrect.sum().item()
                vlen     = vlen + seqcorrect.nelement()
            for i,(x,y) in enumerate(valloader2):
                xdata       = x.cuda()
                ydata2      = y.cuda()
                output      = model(xdata)
                # xdata <- masked index
                # ydata2 <- answer 
                loss        = F.cross_entropy(output, ydata2, reduction='none')
                vloss2       = vloss2 + loss.mean().item()
                bits2 += (loss).sum().item()

                pred2       = output.argmax(axis=1)
                seqcorrect  = (pred2==ydata2).prod(-1).prod(-1)
                vcorrect2 = vcorrect2 + seqcorrect.sum().item()
                vlen2     = vlen2 + seqcorrect.nelement()
            for i,(x,y) in enumerate(testloader):
                xdata       = x.cuda()
                ydata2      = y.cuda()
                output      = model(xdata)
                pred2       = output.argmax(axis=1)
                seqcorrect  = (pred2==ydata2).prod(-1).prod(-1)
                tcorrect = tcorrect + seqcorrect.sum().item()
                tlen     = tlen + seqcorrect.nelement()

            
        accuracyResult = list()

        print("\nval accuracy at ID = {}".format(vcorrect/vlen))
        accuracyResult.append("val accuracy at ID = {}".format(vcorrect/vlen))
        print('validation loss:\t{}'.format(vloss/len(valloader)))
        accuracyResult.append('validation loss:\t{}'.format(vloss/len(valloader)))
        #Perplexity  = 2^bit
        print('Perplexity :\t{}'.format(math.exp((bits/vlen) * math.log(2)))) 
        accuracyResult.append('Perplexity :\t{}'.format(math.exp((bits/vlen) * math.log(2))))

        print("\nval accuracy at OOD = {}".format(vcorrect2/vlen2))
        accuracyResult.append("val accuracy at OOD = {}".format(vcorrect2/vlen2))
        print('validation loss:\t{}'.format(vloss2/len(valloader2)))
        accuracyResult.append('validation loss:\t{}'.format(vloss2/len(valloader2)))
        #Perplexity  = 2^bit
        print('Perplexity :\t{}'.format(math.exp((bits2/tlen) * math.log(2)))) 
        accuracyResult.append('Perplexity :\t{}'.format(math.exp((bits2/tlen) * math.log(2))))

        print("\nTest accuracy = {}".format(tcorrect/tlen))
        accuracyResult.append("Test accuracy = {}".format(tcorrect/tlen))
        #Sequence accuracy
        

        return model, accuracyResult, tcorrect/tlen

def logger(args, timestamp, epoch, contents):
    with open(str("log/") + str(args.exp) + " " + str(time.strftime("%Y-%m-%d %H:%M:%S", timestamp)) + " "+ str(args.seq_type) + " " + str(args.net) +".log", "a+") as fd:
        fd.write('\nEpoch #{}:'.format(epoch))
        fd.write('\n')
        # print model information
        if epoch == 0:
            fd.write(contents)
            fd.write('\n')
            return
        # print experiment result
        for sen in contents:
            fd.write(sen)
            fd.write('\n')

if __name__ == '__main__':

    # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
    #torch.backends.cuda.matmul.allow_tf32 = False

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    #torch.backends.cudnn.allow_tf32 = False
    args = Options.get_args()
    #torch.autograd.set_detect_anomaly(args.debug)
    
    dataset     = GuidedListops('listopsdata/basic_train.tsv')
    valset      = GuidedListops('listopsdata/basic_valid.tsv')
    valset2      = GuidedListops('listopsdata/basic_args.tsv')
    testset      = GuidedListops('listopsdata/basic_depth.tsv')



    vocab_size = dataset.vocab_size

    if args.model_size == 'base':
        dmodel = 768
        nhead = 12
        num_layers = 12
    elif args.model_size == 'mini':
        dmodel = 256
        nhead = 4
        num_layers = 4
    elif args.model_size == 'small':
        dmodel = 512
        nhead = 8
        num_layers = 4
    elif args.model_size == 'medium':
        dmodel = 512
        nhead = 8
        num_layers = 8
    elif args.model_size == 'tiny':
        dmodel = 128
        nhead = 2
        num_layers = 2
    elif args.model_size == 'custom':
        dmodel = 512
        nhead = 4
        num_layers = 4
    else:
        print('shouldnt be here')
        exit(-1)


    if args.net == 'tf':
        print('Executing Autoencoder model with Transformer AE Model')
        model = Models.TfAE(dmodel, nhead=nhead, num_layers=num_layers, vocab_size = vocab_size).cuda()
    elif args.net == 'cnn':
        print('Executing Autoencoder model with CNN AE Model')
        model = Models.CNNAE(dmodel, vocab_size = vocab_size).cuda()
    elif args.net == 'xlnet':
        print('Executing Autoencoder model with XLNet-like Model')
        model = Models.XLNetAE(dmodel, vocab_size = vocab_size, num_layers=num_layers, nhead=nhead).cuda()
    elif args.net == 'lstm':
        print('Executing Autoencoder model with LSTM including Attention')
        model = LSTMListops(int(dmodel*math.sqrt(num_layers)), vocab_size = vocab_size).cuda()
    elif args.net == 'dnc':
        print('Executing DNC model')
        #model = Models.DNCAE(dmodel + dmodel//2, nhead, vocab_size=vocab_size).cuda()
        model = Models.DNCMDSAE(dmodel*2, nhead, vocab_size=vocab_size, mem_size=(dmodel*2)//nhead).cuda()
    elif args.net == 'lsam':
        print('Executing LSAM model')
        model = NAM.LSAMAE(dmodel*2, nhead, vocab_size=vocab_size).cuda()
    elif args.net == 'namtm':
        print('Executing NAM-TM model')
        model = NAMTMListops(dmodel*2, vocab_size, nhead=nhead, debug=args.debug, mem_size=(dmodel*2)//nhead).cuda()
    elif args.net in ['namtm2','nojump','onlyjump','norwprob','noerase']:
        print('Executing NAM-TM model')
        model = NAMTMListops2(dmodel*2, vocab_size, nhead=nhead, debug=args.debug, mem_size=(dmodel*2)//nhead).cuda()
    elif args.net == 'ut':
        print('Executing Universal Transformer model')
        #model = Models.UTAE(dmodel*3, nhead=nhead, num_layers=num_layers, vocab_size = vocab_size).cuda()
        model = UTListops(dmodel*3, nhead=nhead, num_layers=num_layers, vocab_size = vocab_size).cuda()
    elif args.net == 'stm':
        print('Executing STM model')
        #model = Models.UTAE(dmodel*3, nhead=nhead, num_layers=num_layers, vocab_size = vocab_size).cuda()
        model = STM.STMAE(dmodel*2, vocab_size, nhead=nhead, mem_size=(dmodel*2)//nhead).cuda()
    else :
        print('Network {} not supported'.format(args.net))
        exit()
    print(args)
    print(model)
    print("Parameter count: {}".format(Options.count_params(model)))
    col_fn = None
    trainloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=col_fn)
    valloader   = DataLoader(valset, batch_size=args.batch_size, num_workers=4, collate_fn=col_fn)
    valloader2   = DataLoader(valset2, batch_size=args.batch_size, num_workers=4, collate_fn=col_fn)
    testloader   = DataLoader(testset, batch_size=args.batch_size, num_workers=4, collate_fn=col_fn)
    optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lr/10)
    scheduler   = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.98)
    criterion   = nn.CrossEntropyLoss(reduction='none')
    nsamples = len(dataset)
    #torch.autograd.set_detect_anomaly(True)
    if args.log:
        ts = time.gmtime()
        logger(args, ts, 0, str(args))
        logger(args, ts, 0, args.logmsg)
        logger(args, ts, 0, str(model))
        logger(args, ts, 0, "Parameter count: {}".format(Options.count_params(model)))
    
    bestacc = -0.1
    for e in range(args.epochs):
        print('\nEpoch #{}:'.format(e+1))
        if e == 3:
            e = 3#this is the debug point
        trainstart = time.time()
        #train the model
        model, trainResult = train(model, trainloader, criterion, optimizer, scheduler)
        print("Train sequences per second : " + str(nsamples/(time.time()-trainstart)))
        trainResult.append("Train sequences per second : " + str(nsamples/(time.time()-trainstart)))

        #validate the model
        model, valResult, testAcc = validate(model, valloader, valloader2, testloader, args)
        
        if args.log:
            if testAcc > bestacc:
                print("Current best found. Saving pth")
                bestacc=testAcc
                pthfile = str("log/") + str(args.exp) + "_" + \
                    str(time.strftime("%Y-%m-%d %H:%M:%S", ts)) + "_"+ 'guidelistops' + \
                    "_" + str(args.net) + "_" + args.model_size +".pth", "w"
                #torch.save(model.state_dict(), pthfile)
            #save into logfile
            trainResult.extend(valResult)
            logger(args, ts, e+1, trainResult)

    print('Done')
