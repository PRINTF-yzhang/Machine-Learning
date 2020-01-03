import matplotlib.pyplot as plt
import numpy as np
test_acc_path = 'result/hidden_50_lr_0.05_batch_16_epoch_100_test_acc.npy'
test_acc = np.load(test_acc_path)
test_loss_path = 'result/hidden_50_lr_0.05_batch_16_epoch_100_validation_loss.npy'
test_loss = np.load(test_loss_path)

test_acc =np.array(test_acc)
test_loss =np.array(test_loss)

plt.title('test loss and accuracy')
plt.plot(test_acc[:,0],test_acc[:,2],color='red',label='test accuracy')
plt.plot(test_loss[:,0],test_loss[:,2],color='blue',label='test loss')
plt.legend()
plt.xlabel('epoch')
plt.savefig('img/test_loss_acc.png')