import numpy as np
import os
import matplotlib.pyplot as plt


def findBestHyperparameters():
    result_dir = 'result/'
    result_list = os.listdir(result_dir)
    train_loss = {}
    val_loss = {}
    best_loss = 100
    best_name = None

    for path in result_list:
        print(path)
        path_name = os.path.join(result_dir,path)
        result = np.load(path_name)
        result = np.array(result)
        if path.split('_')[-2] == 'train':
            train_loss[path.split('_train_loss.npy')[0]] = result
        if path.split('_')[-2] == 'validation':
            val_loss[path.split('_validation_loss.npy')[0]] = result
        model_name = path.split('_loss.npy')[0]
        loss = result[:,2]
        min_loss = min(loss)
        if model_name.split('_')[-1] == 'validation' and min_loss < best_loss:
            best_loss = min_loss
            best_name = model_name

    print("*"*80)
    print('The best hyperparameters: ', best_name)
    print('The best validation loss:', best_loss)

    colors = ['b','g','r','c','m','y','k','w','skyblue','b','g','r','c','m','y']

    plt.title('training loss')
    idx = 0
    for k in sorted(train_loss.keys()):
        plt.plot(train_loss[k][:,0],train_loss[k][:,2],color=colors[idx],label=k)
        # print(k)
        idx += 1
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    #plt.show()
    plt.savefig('img/train_loss.png')

    plt.cla()
    # print("*" *80)
    plt.title('validation loss')
    idx = 0
    for k in sorted(train_loss.keys()):
        plt.plot(val_loss[k][:,0],val_loss[k][:,2],color=colors[idx],label=k)
        # print(k)
        idx += 1
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    #plt.show()
    plt.savefig('img/val_loss.png')


if __name__ == '__main__':
    findBestHyperparameters()
