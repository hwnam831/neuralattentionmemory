import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import Options
import Models
from NSPDataset import ReductionDatasetAE, NSPDatasetAE, NSPDatasetV2, StringDataset,RepeatedCopy, fib, arith
from ListOpsDataset import ListopsDataset
import STM
#from SCANDataset import SCANResplitAE
import NAM
from torch.utils.data import DataLoader
import time
import math
from DYCK import DYCKDataset
from StackRNN import StackRNNAE

def train(model, trainloader, criterion, optimizer, scheduler):
        model.train(mode=True)
        tcorrect = 0
        tlen     = 0
        tloss    = 0
        bits = 0.0
        maskcount = 0
        for x,y in trainloader:
            xdata       = x.cuda()
            ydata       = y.cuda()
            optimizer.zero_grad()
            output      = model(xdata)

            ismask = xdata != ydata
            maskcount += ismask.sum().item()

            loss        = criterion(output, ydata)
            loss.mean().backward()
            bits += (loss*ismask).sum().item()

            tloss       = tloss + loss.mean().item()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            
            pred        = output.argmax(axis=1)
            seqcorrect  = (pred==ydata).prod(-1)
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
        print('Training Perplexity :\t{}'.format(math.exp((bits/maskcount) * math.log(2)))) 
        trainingResult.append('Training Perplexity :\t{}'.format(math.exp((bits/maskcount) * math.log(2))))
       


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
        maskcount = 0
        bits2 = 0.0
        maskcount2 = 0
        tcorrect = 0
        tlen = 0
        with torch.no_grad():
            for i,(x,y) in enumerate(valloader):
                xdata       = x.cuda()
                ydata2      = y.cuda()
                output      = model(xdata)
                # xdata <- masked index
                # ydata2 <- answer 
                ismask = xdata != ydata2
                mcnt = ismask.sum().item()
                loss        = F.cross_entropy(output, ydata2, reduction='none')
                vloss       = vloss + loss.mean().item()
                bits += (loss*ismask).sum().item()
                maskcount += mcnt
                pred2       = output.argmax(axis=1)
                seqcorrect  = (pred2==ydata2).prod(-1)
                vcorrect = vcorrect + seqcorrect.sum().item()
                vlen     = vlen + seqcorrect.nelement()
            for i,(x,y) in enumerate(valloader2):
                xdata       = x.cuda()
                ydata2      = y.cuda()
                output      = model(xdata)
                # xdata <- masked index
                # ydata2 <- answer 
                ismask = xdata != ydata2
                mcnt = ismask.sum().item()
                loss        = F.cross_entropy(output, ydata2, reduction='none')
                vloss2       = vloss2 + loss.mean().item()
                bits2 += (loss*ismask).sum().item()
                maskcount2 += mcnt
                pred2       = output.argmax(axis=1)
                seqcorrect  = (pred2==ydata2).prod(-1)
                vcorrect2 = vcorrect2 + seqcorrect.sum().item()
                vlen2     = vlen2 + seqcorrect.nelement()
            for i,(x,y) in enumerate(testloader):
                xdata       = x.cuda()
                ydata2      = y.cuda()
                output      = model(xdata)
                pred2       = output.argmax(axis=1)
                seqcorrect  = (pred2==ydata2).prod(-1)
                tcorrect = tcorrect + seqcorrect.sum().item()
                tlen     = tlen + seqcorrect.nelement()

            
        accuracyResult = list()

        print("\nval accuracy at ID = {}".format(vcorrect/vlen))
        accuracyResult.append("val accuracy at ID = {}".format(vcorrect/vlen))
        print('validation loss:\t{}'.format(vloss/len(valloader)))
        accuracyResult.append('validation loss:\t{}'.format(vloss/len(valloader)))
        #Perplexity  = 2^bit
        print('Perplexity :\t{}'.format(math.exp((bits/maskcount) * math.log(2)))) 
        accuracyResult.append('Perplexity :\t{}'.format(math.exp((bits/maskcount) * math.log(2))))

        print("\nval accuracy at OOD = {}".format(vcorrect2/vlen2))
        accuracyResult.append("val accuracy at OOD = {}".format(vcorrect2/vlen2))
        print('validation loss:\t{}'.format(vloss2/len(valloader2)))
        accuracyResult.append('validation loss:\t{}'.format(vloss2/len(valloader2)))
        #Perplexity  = 2^bit
        print('Perplexity :\t{}'.format(math.exp((bits2/maskcount2) * math.log(2)))) 
        accuracyResult.append('Perplexity :\t{}'.format(math.exp((bits2/maskcount2) * math.log(2))))

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
    if args.seq_type == 'fib':
        
        dataset     = NSPDatasetAE(fib, args.digits, size=args.train_size)
        valset      = NSPDatasetAE(fib, args.digits, args.digits//2, size=args.validation_size)
        valset2      = NSPDatasetAE(fib, args.digits+6, args.digits+1, size=args.validation_size)
        testset      = NSPDatasetAE(fib, args.digits+12, args.digits+7, size=args.validation_size)
        '''
        dataset     = NSPDatasetV2('fib', args.digits, size=args.train_size)
        valset      = NSPDatasetV2('fib', args.digits, args.digits//2, size=args.validation_size)
        valset2      = NSPDatasetV2('fib', args.digits+6, args.digits+1, size=args.validation_size)
        testset      = NSPDatasetV2('fib', args.digits+12, args.digits+7, size=args.validation_size)
        '''
    elif args.seq_type == 'arith':
        dataset     = NSPDatasetAE(arith, args.digits, size=args.train_size)
        valset      = NSPDatasetAE(arith, args.digits, args.digits//2, size=args.validation_size)
        valset2      = NSPDatasetAE(arith, args.digits+6, args.digits+1, size=args.validation_size)
        testset      = NSPDatasetAE(arith, args.digits+12, args.digits+7, size=args.validation_size)
    elif args.seq_type == 'copy':
        dataset     = RepeatedCopy(3, args.digits, size=args.train_size)
        valset      = RepeatedCopy(3, args.digits, args.digits//2, size=args.validation_size)
        valset2      = RepeatedCopy(3, args.digits+6, args.digits+1, size=args.validation_size)
        testset      = RepeatedCopy(4, args.digits+12, args.digits+7, size=args.validation_size)
    elif args.seq_type == 'palin':
        dataset     = StringDataset(args.seq_type, args.digits, size=args.train_size)
        valset      = StringDataset(args.seq_type, args.digits, args.digits//2, size=args.validation_size)
        valset2      = StringDataset(args.seq_type, args.digits+6, args.digits+1, size=args.validation_size)
        testset      = StringDataset(args.seq_type, args.digits+12, args.digits+7, size=args.validation_size)
    elif args.seq_type == 'scan':
        dataset     = SCANResplitAE('train', (0,args.digits))
        valset      = SCANResplitAE('test', (0,args.digits))
        valset2      = SCANResplitAE('all', (args.digits+1,9999))
        testset      = valset2
    elif args.seq_type == 'reduce':
        dataset     = ReductionDatasetAE(args.digits, size=args.train_size)
        valset      = ReductionDatasetAE(args.digits,args.digits//2, size=args.validation_size)
        valset2      = ReductionDatasetAE(args.digits+6,args.digits+1, size=args.validation_size) 
        testset      = ReductionDatasetAE(args.digits+12, args.digits+7, size=args.validation_size) 
    elif args.seq_type == 'listops':
        dataset     = ListopsDataset('listopsdata/basic_train.tsv')
        valset      = ListopsDataset('listopsdata/basic_valid.tsv')
        valset2      = ListopsDataset('listopsdata/basic_args.tsv')
        testset      = ListopsDataset('listopsdata/basic_depth.tsv')
    elif args.seq_type == 'dyck':
        dataset     = DYCKDataset('dyckdata/dyck_train.txt')
        valset      = DYCKDataset('dyckdata/dyck_val.txt')
        valset2      = DYCKDataset('dyckdata/dyck_length.txt')
        testset      = DYCKDataset('dyckdata/dyck_test.txt')


    if args.seq_type == 'scan': 
        vocab_size = dataset.vocab_size
        dictionary = dataset.wordtoix
    elif args.seq_type == 'reduce': 
        vocab_size = dataset.vocab_size
    elif args.seq_type == 'listops': 
        vocab_size = dataset.vocab_size
    else:
        vocab_size = 16

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
        model = Models.LSTMAE(int(dmodel*math.sqrt(num_layers)), vocab_size = vocab_size).cuda()
    elif args.net == 'noatt':
        print('Executing Autoencoder model with LSTM w.o. Attention')
        model = Models.LSTMNoAtt(int(dmodel*math.sqrt(num_layers)), vocab_size = vocab_size).cuda()
    elif args.net == 'dnc':
        print('Executing DNC model')
        #model = Models.DNCAE(dmodel + dmodel//2, nhead, vocab_size=vocab_size).cuda()
        model = Models.DNCMDSAE(dmodel*2, nhead, vocab_size=vocab_size, mem_size=(dmodel*2)//nhead).cuda()
    elif args.net == 'lsam':
        print('Executing LSAM model')
        model = NAM.LSAMAE(dmodel*2, nhead, vocab_size=vocab_size).cuda()
    elif args.net == 'namtm':
        print('Executing NAM-TM model')
        model = NAM.NAMTMAE(dmodel*2, vocab_size, nhead=nhead, debug=args.debug, mem_size=(dmodel*2)//nhead).cuda()
    elif args.net in ['namtm2','nojump','onlyjump','norwprob','noerase']:
        print('Executing NAM-TM model')
        model = NAM.NAMTMAE(dmodel*2, vocab_size, nhead=nhead, mem_size=(dmodel*2)//nhead, option=args.net, debug=args.debug).cuda()
    elif args.net == 'ut':
        print('Executing Universal Transformer model')
        #model = Models.UTAE(dmodel*3, nhead=nhead, num_layers=num_layers, vocab_size = vocab_size).cuda()
        model = Models.UTRelAE(dmodel*3, nhead=nhead, num_layers=num_layers, vocab_size = vocab_size).cuda()
    elif args.net == 'stm':
        print('Executing STM model')
        #model = Models.UTAE(dmodel*3, nhead=nhead, num_layers=num_layers, vocab_size = vocab_size).cuda()
        model = STM.STMAE(dmodel*2, vocab_size, nhead=nhead, mem_size=(dmodel*2)//nhead).cuda()
    elif args.net == 'stack':
        print('Executing Stack RNN model')
        #model = Models.UTAE(dmodel*3, nhead=nhead, num_layers=num_layers, vocab_size = vocab_size).cuda()
        model = StackRNNAE(dmodel*4, vocab_size=vocab_size, nhead=nhead, mem_size=(dmodel*4)//nhead).cuda()
    else :
        print('Network {} not supported'.format(args.net))
        exit()
    print(args)
    print(model)
    print("Parameter count: {}".format(Options.count_params(model)))
    col_fn = SCANResplitAE.collate_batch if args.seq_type == 'scan' else None
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
                    str(time.strftime("%Y-%m-%d %H:%M:%S", ts)) + "_"+ str(args.seq_type) + \
                    "_" + str(args.net) + "_" + args.model_size +".pth", "w"
                #torch.save(model.state_dict(), pthfile)
            #save into logfile
            trainResult.extend(valResult)
            logger(args, ts, e+1, trainResult)

    print('Done')
