import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes

def save_result(obj, name ):
    with open(''+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_result(name ):
    with open('' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
VGG_Net = load_result("VGG_performances_result_final")
EPOCHS = 150
iter_for_epoch = 50000//128 + 1 

fig = plt.figure(figsize=(15, 12))

ax = fig.add_subplot(221)

plt.plot(VGG_Net['Adam']['HD']['alpha_epoch'],
             color='indigo', linestyle='-', label = 'Adam-HD')
plt.plot([0,EPOCHS],[0.001,0.001], color ='tab:purple',
         linestyle = "--", label = "Adam")
plt.plot(VGG_Net['SGDN']['HD']['alpha_epoch'],
             color='mediumvioletred', linestyle='-', label="SGDN-HD")
plt.plot([0,EPOCHS],[0.001,0.001], color ='palevioletred',
         linestyle = "--", label = "SGDN")
plt.plot(VGG_Net['SGD']['HD']['alpha_epoch'],
             color='tab:orange', linestyle='-', label ="SGD-HD")

plt.plot([0,EPOCHS],[0.001,0.001], color ='goldenrod',
         linestyle = "--", label = "SGD")
plt.ylabel(r'$\alpha_t$')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid()



ax = fig.add_subplot(222)
plt.plot(range(1, EPOCHS*iter_for_epoch + 1), VGG_Net['Adam']['HD']['alpha_it'],
             color='indigo')
plt.plot(range(1, EPOCHS*iter_for_epoch + 1), VGG_Net['SGDN']['HD']['alpha_it'],
             color='mediumvioletred')
plt.plot(range(1, EPOCHS*iter_for_epoch + 1), VGG_Net['SGD']['HD']['alpha_it'],
             color='tab:orange')
plt.xlabel('Iteration')
plt.ylabel(r'$\alpha_t$')
plt.xscale('log')
plt.legend(loc='upper right')
plt.plot([1,EPOCHS*iter_for_epoch],[0.001,0.001], color ='tab:purple',
         linestyle = "--")
plt.grid()

ax = fig.add_subplot(223)
plt.plot(VGG_Net['Adam']['HD']['loss'], linestyle='-', color='indigo')
plt.plot(VGG_Net['Adam']['Keras']['loss'], linestyle='--', color='tab:purple')
plt.plot(VGG_Net['SGDN']['HD']['loss'], linestyle='-', color='mediumvioletred')
plt.plot(VGG_Net['SGDN']['Keras']['loss'], linestyle='--', color='palevioletred')
plt.plot(VGG_Net['SGD']['HD']['loss'], linestyle='-', color='tab:orange')
plt.plot(VGG_Net['SGD']['Keras']['loss'], linestyle='--', color='goldenrod')
plt.xlabel('Epoch')
plt.ylim((1e-2-1e-3,20))
plt.yscale('log')
plt.grid()
inset_axes(ax, width="50%", height="35%", loc=1)
plt.plot(range(1, EPOCHS*iter_for_epoch + 1), VGG_Net['Adam']['HD']['loss_it'],
             color='indigo')
plt.plot(range(1, EPOCHS*iter_for_epoch + 1), VGG_Net['Adam']['Keras']['loss_it'],
            color='tab:purple', linestyle = "--")
plt.plot(range(1, EPOCHS*iter_for_epoch + 1), VGG_Net['SGDN']['HD']['loss_it'],
             color='mediumvioletred')
plt.plot(range(1, EPOCHS*iter_for_epoch + 1), VGG_Net['SGDN']['Keras']['loss_it'],
             color='palevioletred', linestyle='--')
plt.plot(range(1, EPOCHS*iter_for_epoch + 1), VGG_Net['SGD']['HD']['loss_it'],
             color='tab:orange')
plt.plot(range(1, EPOCHS*iter_for_epoch + 1), VGG_Net['SGD']['Keras']['loss_it'],
             color='goldenrod', linestyle='--')
plt.xlabel('Iteration')
plt.ylabel('Train loss')
plt.xscale('log')
plt.grid()

ax = fig.add_subplot(224)

plt.plot(VGG_Net['Adam']['HD']['val_loss'],
             color='indigo', label=r'SGDM HD $\alpha_0=10^{-1}$')
plt.plot(VGG_Net['Adam']['Keras']['val_loss'], linestyle = "--",
             color='tab:purple', label=r'SGDM HD $\alpha_0=10^{-1}$') 
plt.plot(VGG_Net['SGDN']['HD']['val_loss'], 
             color='mediumvioletred', label=r'SGDM HD $\alpha_0=10^{-1}$')
plt.plot(VGG_Net['SGDN']['Keras']['val_loss'], linestyle = "--", 
             color='palevioletred', label=r'SGDM HD $\alpha_0=10^{-1}$')
plt.plot(VGG_Net['SGD']['HD']['val_loss'], 
             color='tab:orange', label=r'SGDM HD $\alpha_0=10^{-1}$')
plt.plot(VGG_Net['SGD']['Keras']['val_loss'], linestyle = "--", 
             color='goldenrod', label=r'SGDM HD $\alpha_0=10^{-1}$')
plt.grid() 
plt.yscale('log')
plt.show()