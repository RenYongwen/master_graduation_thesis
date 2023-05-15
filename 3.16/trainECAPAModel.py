'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''

import argparse
import glob
import os
import torch
import warnings
import time
import sys
from dataLoader import DataLoader
from model import *


parser = argparse.ArgumentParser(description="ECAPA_trainer")
# Training Settings
parser.add_argument('--num_frames', type=int, default=200, help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch', type=int, default=100, help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--n_cpu', type=int, default=8, help='Number of loader threads')
parser.add_argument('--test_step', type=int, default=1, help='Test and save every [test_step] epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument("--lr_decay", type=float, default=0.966, help='Learning rate decay every [test_step] epochs')

# Training and evaluation path/lists, save path
parser.add_argument('--data_path', type=str, default="common_voice_kpd", help='The path of the training data')
parser.add_argument('--musan_path', type=str, default="musan", help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')
parser.add_argument('--rir_path', type=str, default="RIRS_NOISES", help='The path to the RIR set, eg:"/data08/Others/RIRS_NOISES/simulated_rirs" in my case')
parser.add_argument('--save_path', type=str, default="./output", help='Path to save the score.txt and models')
parser.add_argument('--initial_model', type=str, default="", help='Path of the initial_model')

# Model and Loss settings
parser.add_argument('--C', type=int, default=1024, help='Channel size for the speaker encoder')
parser.add_argument('--margin', type=float, default=0.2, help='Loss margin in AAM softmax')
parser.add_argument('--scale', type=float, default=30, help='Loss scale in AAM softmax')
parser.add_argument('--n_class', type=int, default=45, help='Number of speakers')

# Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()


def train_network(model, epoch, loader):
    model.train()
    # Update the learning rate based on the current epcoh
    scheduler.step(epoch - 1)
    index, top1, loss = 0, 0, 0
    lr = optimzer.param_groups[0]['lr']
    for num, (data, labels) in enumerate(loader, start=1):
        model.zero_grad()
        labels = torch.LongTensor(labels).cuda()
        embedding = model.forward(data.cuda(), aug=True)
        output = classifier.forward(embedding)
        nloss,prec = losser.forward(output, labels)
        nloss.backward()
        optimzer.step()
        # Print information
        index += len(labels)
        top1 += prec
        loss += nloss.detach().cpu().numpy()
        sys.stderr.write(time.strftime("%m-%d %H:%M:%S") +
                         " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / loader.__len__())) +
                         " Loss: %.5f, Prec: %2.2f%% \r" % (loss / (num), top1 / index * len(labels)))
        sys.stderr.flush()
    sys.stdout.write("\n")
    return loss / num, lr


def eval_network(model, loader):
    model.eval()  # 这句话也是必须的
    with torch.no_grad():
        total_correct = 0
        total_num = 0
        for x, labels in loader:
            x, labels = x.cuda(), labels.cuda()
            embedding = model.forward(x, aug=True)
            output = classifier.forward(embedding)
            nloss, prec = losser.forward(output, labels)
            total_correct += x.size(0) * prec
            total_num += x.size(0)
        acc = total_correct / total_num
    return acc


def save_models(model):
    traced_model = torch.jit.trace(
        model, torch.FloatTensor(torch.randn(256, 32240)).cuda())
    traced_model.save(args.save_path + "/model_%04d.pt" % epoch)


# Define the data loader
trainloader = DataLoader(**vars(args))
trainLoader = torch.utils.data.DataLoader(trainloader, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=True)
evalloader = DataLoader(train=False, **vars(args))
evalLoader = torch.utils.data.DataLoader(evalloader, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=False)


# Search for the exist models
modelfiles = glob.glob('%s/model_0*.model' % args.save_path)
modelfiles.sort()
score_file = open(os.path.join(args.save_path, "score.txt"), "a+")

# ECAPA-TDNN
if args.initial_model != "":
    print("Model %s loaded from previous state!" % args.initial_model)
    model = ECAPA_TDNN(**vars(args)).cuda()
    model.load_state_dict(torch.load(args.initial_model))
    epoch = 1
elif len(modelfiles) >= 1:
    print("Model %s loaded from previous state!" % modelfiles[-1])
    epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
    model = ECAPA_TDNN(**vars(args)).cuda()
    model.load_state_dict(torch.load(modelfiles[-1]))
    save_models(model)
else:
    epoch = 1
    model = ECAPA_TDNN(**vars(args)).cuda()

# Classifier
classifier = Classifier(input_size=192, out_neurons=45).cuda()

loss_fn = AdditiveAngularMargin(**vars(args)).cuda()
losser = LogSoftmaxWrapper(loss_fn=loss_fn).cuda()

# losser = LogAAMSoftmaxWrapper(**vars(args)).cuda()
optimzer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=2e-6)
scheduler = torch.optim.lr_scheduler.StepLR(optimzer, step_size=args.test_step, gamma=args.lr_decay)
print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" % (sum(param.numel() for param in model.parameters()) / 1024 / 1024))
lr_list = torch.linspace(args.lr,args.lr/30,args.max_epoch).tolist()
if __name__ == '__main__':
    while (1):
        # Training for one epoch
        loss, lr = train_network(model, epoch, trainLoader)

        # Evaluation every [test_step] epochs
        if epoch % args.test_step == 0:
            torch.save(model.state_dict(), args.save_path + "/model_%04d.model" % epoch)
            acc = eval_network(model, evalLoader)
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, Lr:%f, Loss:%f, ACC:%2.2f%%" % (epoch, lr, loss, acc))
            score_file.write("%d epoch, Lr:%f, Loss:%f, Acc:%2.2f%%\n" % (epoch, lr, loss, acc))
            score_file.flush()

        if epoch >= args.max_epoch:
            quit()

        epoch += 1
