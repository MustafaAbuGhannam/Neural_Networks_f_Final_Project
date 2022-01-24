import matplotlib.pyplot as plt
import torch
from Net import Net
from contrastiveLoss import ContrastiveLoss
from train import train
from test import test
from data_set import LinesDataSet
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    writer_ = SummaryWriter('runs/without_reg_for_160_writers_on_english')
    train_acc = []
    test_acc = []
    model = Net()
    epochs = 30
    loss_function = ContrastiveLoss()
    loss_history = []
    train_line_data_set = LinesDataSet(csv_file="Train_labels_for_english.csv", root_dir="english_data_set", transform=transforms.Compose([transforms.ToTensor()]))
    test_line_data_set = LinesDataSet(csv_file="Test_labels_for_english.csv", root_dir='english_data_set', transform=transforms.Compose([transforms.ToTensor()]))
    train_line_data_loader = DataLoader(train_line_data_set,shuffle=True,batch_size=17)
    test_line_data_loader = DataLoader(test_line_data_set, shuffle=True, batch_size=1)
    train_line_data_loader_for_test = DataLoader(train_line_data_set,shuffle=True,batch_size=1)

    example = iter(train_line_data_loader)
    example_img1, example_img2, target = example.next()
    
    example_img1 = example_img1[:,None,:,:].float().cuda()
    example_img2 = example_img2[:,None,:,:].float().cuda()

    torch.manual_seed(17)

    if torch.cuda.is_available():
        model = model.cuda()
        loss_function = loss_function.cuda()
    
    writer_.add_graph(model.cuda(), (example_img1, example_img2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(epochs):

        print('epoch number: {}'.format(i + 1))
        train(model, loss_function, optimizer, train_line_data_loader, loss_history)
        writer_.add_scalar('train_loss', loss_history[i], i)
        print('testing on train data set...')

        y_pred = []
        y_true = []

        test(model, train_line_data_loader_for_test, acc_history= [], all_acc_train=train_acc, all_acc_test= test_acc, train_flag=True, y_pred = y_pred, y_true = y_true)
        writer_.add_scalar('train_acc', train_acc[i], i)
        print('testing on test data set...')

        y_pred = []
        y_true = []

        test(model, train_line_data_loader_for_test, acc_history= [], all_acc_train=train_acc, all_acc_test= test_acc, train_flag=False, y_pred = y_pred, y_true = y_true)
        writer_.add_scalar('test_acc', test_acc[i], i)

        print('creating confusion_matrix...')
        cf_matrix = confusion_matrix(y_true, y_pred)
        classes = ('0', '1')
        df_cm = pd.DataFrame(cf_matrix / 4000, index = [i for i in classes],
                        columns = [i for i in classes])
        plt.figure(figsize = (12,7))

        writer_.add_figure('confusion_matrix_without_reg', sn.heatmap(df_cm, annot=True).get_figure(), i)
        
        writer_.close() 

