'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''

import argparse, glob, os, torch, warnings, time, sys
from dataLoader import train_loader,eval_loader
import torch.nn as nn
from model import ECAPA_TDNN



parser = argparse.ArgumentParser(description="ECAPA_trainer")
## Training Settings	
parser.add_argument('--num_frames', type=int, default=200, help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch', type=int, default=40, help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--n_cpu', type=int, default=4, help='Number of loader threads')
parser.add_argument('--test_step', type=int, default=1, help='Test and save every [test_step] epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument("--lr_decay", type=float, default=0.97, help='Learning rate decay every [test_step] epochs')

## Training and evaluation path/lists, save path
parser.add_argument('--data_path', type=str, default="common_voice_kpd", help='The path of the training data')
parser.add_argument('--musan_path', type=str, default="musan", help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')
parser.add_argument('--rir_path', type=str, default="RIRS_NOISES", help='The path to the RIR set, eg:"/data08/Others/RIRS_NOISES/simulated_rirs" in my case');
parser.add_argument('--model_save_path', type=str, default="../output/model", help='Path to save the score.txt and models')
parser.add_argument('--score_save_path', type=str, default="../output/score", help='Path to save the score.txt and models')
parser.add_argument('--initial_model', type=str, default="", help='Path of the initial_model')

## Model and Loss settings
parser.add_argument('--C', type=int, default=1024, help='Channel size for the speaker encoder')
parser.add_argument('--m', type=float, default=0.2, help='Loss margin in AAM softmax')
parser.add_argument('--s', type=float, default=30, help='Loss scale in AAM softmax')
parser.add_argument('--n_class', type=int, default=45, help='Number of speakers')

## Command
parser.add_argument('--eval', dest='eval', action='store_true', help='Only do evaluation')

## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()






def train_network(model, epoch, loader):
    model.train()
    ## Update the learning rate based on the current epcoh
    scheduler.step(epoch - 1)
    index, loss = 0, 0
    lr = optim.param_groups[0]['lr']
    for num, (data, labels) in enumerate(loader, start = 1):
        model.zero_grad()
        labels            = torch.LongTensor(labels).cuda()
        speaker_embedding = model.forward(data.cuda(), aug = True)
        nloss             = speaker_loss(speaker_embedding, labels)
        nloss.backward()
        optim.step()
        # Print information
        index += len(labels)
        loss += nloss.detach().cpu().numpy()
        sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
        " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
        " Loss: %.5f \r"        %(loss/(num)))
        sys.stderr.flush()
    sys.stdout.write("\n")
    return loss/num, lr

def eval_network(model, loader):
    model.eval()    #这句话也是必须的
    with torch.no_grad():
        total_correct = 0
        total_num = 0
        for x, label in loader:
            x, label = x.cuda(), label.cuda()
            logits = model.forward(x, aug = False)
            pred = logits.argmax(dim=1)
            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)
        acc = total_correct / total_num
    return acc*100


def save_parameters(model, path):
    torch.save(model.state_dict(), path)

def load_parameters(model, path):
    self_state = model.state_dict()
    loaded_state = torch.load(path)
    for name, param in loaded_state.items():
        origname = name
        if name not in self_state:
            name = name.replace("module.", "")
            if name not in self_state:
                print("%s is not in the model."%origname)
                continue
        if self_state[name].size() != loaded_state[origname].size():
            print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
            continue
        self_state[name].copy_(param)



## Define the data loader
trainloader = train_loader(**vars(args))
trainLoader = torch.utils.data.DataLoader(trainloader, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=True)
evalloader = eval_loader(**vars(args))
evalLoader = torch.utils.data.DataLoader(evalloader, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=True)


## Search for the exist models
score_file = open(args.score_save_path, "a+")
modelfiles = glob.glob('%s/model_0*.model' % args.model_save_path)
modelfiles.sort()

## Only do evaluation, the initial_model is necessary
if args.eval == True:
    model = ECAPA_TDNN(**vars(args)).cuda()
    print("Model %s loaded from previous state!" % args.initial_model)
    load_parameters(model, args.initial_model)
    eval_network(model,data_path=args.data_path)
    quit()

## ECAPA-TDNN
if args.initial_model != "":
    print("Model %s loaded from previous state!" % args.initial_model)
    model = ECAPA_TDNN(**vars(args)).cuda()
    load_parameters(model, args.initial_model)
    epoch = 1
elif len(modelfiles) >= 1:
    print("Model %s loaded from previous state!" % modelfiles[-1])
    epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
    model = ECAPA_TDNN(**vars(args)).cuda()
    load_parameters(model, modelfiles[-1])
else:
    epoch = 1
    model = ECAPA_TDNN(**vars(args)).cuda()
    
## Classifier
speaker_loss    = nn.CrossEntropyLoss().cuda()
optim           = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = 2e-6)
scheduler       = torch.optim.lr_scheduler.StepLR(optim, step_size = args.test_step, gamma=args.lr_decay)
print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in model.parameters()) / 1024 / 1024))






if __name__ == '__main__':
    while (1):
        ## Training for one epoch
        loss, lr, acc = train_network(model, epoch, trainLoader)

        ## Evaluation every [test_step] epochs
        if epoch % args.test_step == 0:
            save_parameters(model,args.model_save_path + "/model_%04d.model" % epoch)
            acc = eval_network(model,evalLoader)
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, LR %f,LOSS %f, ACC %2.2f%%" % (epoch, lr, loss,acc))
            score_file.write(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, LR %f,LOSS %f, ACC %2.2f%%" % (epoch, lr, loss,acc))
            score_file.flush()

        if epoch >= args.max_epoch:
            quit()

        epoch += 1
