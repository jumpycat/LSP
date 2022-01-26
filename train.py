import torch.optim as optim
from model import *
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

EPOCH = 50
BATCH_SIZE = 32
LENGTH = BATCH_SIZE * 200

net = resnet18().to(device)
pretext_model = torch.load(r'resnet18-5c106cde.pth')
model2_dict = net.state_dict()
state_dict = {k: v for k, v in pretext_model.items() if k in model2_dict.keys()}
state_dict.pop('fc.weight')
state_dict.pop('fc.bias')
model2_dict.update(state_dict)
net.load_state_dict(model2_dict)

net.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters())



if __name__ == '__main__':
    TYPE = 'raw'
    VAL_REAL_ROOT = r'/01-Images/00-FF++/Real/' + TYPE + '/val/'
    VAL_FAKE_ROOT1 = r'/01-Images/00-FF++/Deepfakes/' + TYPE + '/val/'
    VAL_FAKE_ROOT2 = r'/01-Images/00-FF++/Face2Face/' + TYPE + '/val/'
    VAL_FAKE_ROOT3 = r'/01-Images/00-FF++/FaceSwap/' + TYPE + '/val/'
    VAL_FAKE_ROOT4 = r'/01-Images/00-FF++/NeuralTextures/' + TYPE + '/val/'


    TRAIN_FAKE_ROOT = r'/01-Images/00-FF++/Deepfakes/'+TYPE+'/train/'
    TRAIN_REAL_ROOT = r'/01-Images/00-FF++/Real/'+TYPE+'/train/'
    dealDataset = DealDataset(TRAIN_FAKE_ROOT=TRAIN_FAKE_ROOT, TRAIN_REAL_ROOT=TRAIN_REAL_ROOT, LENGTH=LENGTH,TYPE=TYPE)
    train_loader = DataLoader(dataset=dealDataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        print('-'*80)
        net.train()
        step = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, label1, label2, label3, label4,label5 = inputs.to(device), labels[0].to(device), labels[1].to(device), \
                                                     labels[2].to(device), labels[3].to(device),labels[4].to(device)
            optimizer.zero_grad()
            output1, output2, output3, output4,output5 = net(inputs)

            loss1 = criterion(output1, label1)
            loss2 = criterion(output2, label2)
            loss3 = criterion(output3, label3)
            loss4 = criterion(output4, label4)
            loss5 = criterion(output5, label5.float())

            loss = loss1 + loss2 + loss3 + loss4 + loss5
            loss.backward()
            optimizer.step()

            acc = torch.sum(torch.eq(torch.ge(torch.sigmoid(output5), torch.full_like(output5, 0.5)), label5))
            data = '[epoch:%03d, iter:%03d] Loss: %.04f Acc: %.04f' % (epoch + 1, i, loss.item(), acc.cpu().numpy() / BATCH_SIZE)
            with open('-training-logs.txt', 'a', encoding='utf-8') as f:
                f.write(data)
                f.write('\n')
            print(data)
            step += 1

        acc_df = Val(net, VAL_FAKE_ROOT=VAL_FAKE_ROOT1, VAL_REAL_ROOT=VAL_REAL_ROOT)
        acc_f2f = Val(net, VAL_FAKE_ROOT=VAL_FAKE_ROOT2, VAL_REAL_ROOT=VAL_REAL_ROOT)
        acc_fs = Val(net, VAL_FAKE_ROOT=VAL_FAKE_ROOT3, VAL_REAL_ROOT=VAL_REAL_ROOT)
        acc_nt = Val(net, VAL_FAKE_ROOT=VAL_FAKE_ROOT4, VAL_REAL_ROOT=VAL_REAL_ROOT)

        # Deepfakes Face2Face FaceSwap NeuralTextures

        tag = 'epoch-%03d-loss-%.04f-%.04f-%.04f-%.04f-%.04f' % (
                epoch + 1, loss.item(), acc_df, acc_f2f, acc_fs, acc_nt)

        print(tag)

        print('-' * 50)
        with open('-val-logs.txt', 'a', encoding='utf-8') as f:
            f.write(tag)
            f.write('\n')
        torch.save(net, r'models/' + tag + '.pkl')
