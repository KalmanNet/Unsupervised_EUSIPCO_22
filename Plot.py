import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Legend
Klegend = ["Train", "CV", "Test", "Kalman Filter"]
# Color
KColor = ['ro', 'yo', 'g-', 'b-']

class Plot:

    def __init__(self, folderName, modelName):
        self.folderName = folderName
        self.modelName = modelName

    def NNPlot_epochs(self, N_Epochs_plt, MSE_KF_dB_avg,
                      MSE_test_dB_avg, MSE_cv_dB_epoch, MSE_train_dB_epoch,NN_name = ''):

        # File Name
        fileName = self.folderName + 'plt_epochs_dB_'+self.modelName

        fontSize = 30

        # Figure
        plt.figure(figsize = (50, 20))

        # x_axis
        x_plt = range(0, N_Epochs_plt)

        # Train
        y_plt1 = MSE_train_dB_epoch[range(0, N_Epochs_plt)]
        plt.plot(x_plt, y_plt1, KColor[0], label=Klegend[0])

        # CV
        y_plt2 = MSE_cv_dB_epoch[range(0, N_Epochs_plt)]
        plt.plot(x_plt, y_plt2, KColor[1], label=Klegend[1])

        # Test
        y_plt3 = MSE_test_dB_avg * np.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt3, KColor[2], label=Klegend[2])

        # KF
        y_plt4 = MSE_KF_dB_avg * np.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt4, KColor[3], label=Klegend[3])

        plt.legend(prop = {'size':50})
        plt.xlabel('Number of Training Epochs', fontsize=fontSize)
        plt.ylabel('MSE Loss Value [dB]', fontsize=fontSize)
        plt.title(self.modelName + ":" + "MSE Loss [dB] - per Epoch", fontsize=fontSize)
        plt.savefig(fileName+'.eps')


    def NNPlot_Hist(self, MSE_KF_data_linear_arr, MSE_KN_linear_arr,NN_name = ''):

        fileName = self.folderName + 'plt_hist_dB_'+self.modelName

        ####################
        ### dB Histogram ###
        ####################
        plt.figure(figsize=(50, 20))
        sns.distplot(10 * np.log10(MSE_KN_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = self.modelName)
        # sns.distplot(10 * np.log10(MSE_KF_design_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Kalman Filter - design')
        sns.distplot(10 * np.log10(MSE_KF_data_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'r', label = 'Kalman Filter')

        plt.title("Histogram [dB]")
        plt.legend(prop= {'size':50})
        plt.savefig(fileName+'.eps')

    def KFPlot(res_grid):

        plt.figure(figsize = (50, 20))
        x_plt = [-6, 0, 6]

        plt.plot(x_plt, res_grid[0][:], 'xg', label='minus')
        plt.plot(x_plt, res_grid[1][:], 'ob', label='base')
        plt.plot(x_plt, res_grid[2][:], '+r', label='plus')
        plt.plot(x_plt, res_grid[3][:], 'oy', label='base NN')

        plt.legend()
        plt.xlabel('Noise', fontsize=16)
        plt.ylabel('MSE Loss Value [dB]', fontsize=16)
        plt.title('Change', fontsize=16)
        plt.savefig('plt_grid_dB')

        print("\ndistribution 1")
        print("Kalman Filter")
        print(res_grid[0][0], "[dB]", res_grid[1][0], "[dB]", res_grid[2][0], "[dB]")
        print(res_grid[1][0] - res_grid[0][0], "[dB]", res_grid[2][0] - res_grid[1][0], "[dB]")
        print("KalmanNet", res_grid[3][0], "[dB]", "KalmanNet Diff", res_grid[3][0] - res_grid[1][0], "[dB]")

        print("\ndistribution 2")
        print("Kalman Filter")
        print(res_grid[0][1], "[dB]", res_grid[1][1], "[dB]", res_grid[2][1], "[dB]")
        print(res_grid[1][1] - res_grid[0][1], "[dB]", res_grid[2][1] - res_grid[1][1], "[dB]")
        print("KalmanNet", res_grid[3][1], "[dB]", "KalmanNet Diff", res_grid[3][1] - res_grid[1][1], "[dB]")

        print("\ndistribution 3")
        print("Kalman Filter")
        print(res_grid[0][2], "[dB]", res_grid[1][2], "[dB]", res_grid[2][2], "[dB]")
        print(res_grid[1][2] - res_grid[0][2], "[dB]", res_grid[2][2] - res_grid[1][2], "[dB]")
        print("KalmanNet", res_grid[3][2], "[dB]", "KalmanNet Diff", res_grid[3][2] - res_grid[1][2], "[dB]")

    def NNPlot_test(MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg,
               MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg):


        N_Epochs_plt = 100

        ###############################
        ### Plot per epoch [linear] ###
        ###############################
        plt.figure(figsize = (50, 20))

        x_plt = range(0, N_Epochs_plt)

        # KNet - Test
        y_plt3 = MSE_test_linear_avg * np.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt3, KColor[2], label=Klegend[2])

        # KF
        y_plt4 = MSE_KF_linear_avg * np.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt4, KColor[3], label=Klegend[3])

        plt.legend()
        plt.xlabel('Number of Training Epochs', fontsize=16)
        plt.ylabel('MSE Loss Value [linear]', fontsize=16)
        plt.title('MSE Loss [linear] - per Epoch', fontsize=16)
        plt.savefig('plt_model_test_linear')

        ###########################
        ### Plot per epoch [dB] ###
        ###########################
        plt.figure(figsize = (50, 20))

        x_plt = range(0, N_Epochs_plt)

        # KNet - Test
        y_plt3 = MSE_test_dB_avg * np.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt3, KColor[2], label=Klegend[2])

        # KF
        y_plt4 = MSE_KF_dB_avg * np.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt4, KColor[3], label=Klegend[3])

        plt.legend()
        plt.xlabel('Number of Training Epochs', fontsize=16)
        plt.ylabel('MSE Loss Value [dB]', fontsize=16)
        plt.title('MSE Loss [dB] - per Epoch', fontsize=16)
        plt.savefig('plt_model_test_dB')

        ########################
        ### Linear Histogram ###
        ########################
        plt.figure(figsize=(50, 20))
        sns.distplot(MSE_test_linear_arr, hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = 'KalmanNet')
        sns.distplot(MSE_KF_linear_arr, hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Kalman Filter')
        plt.title("Histogram [Linear]")
        plt.savefig('plt_hist_linear')

        fig, axes = plt.subplots(2, 1, figsize=(50, 20), sharey=True, dpi=100)
        sns.distplot(MSE_test_linear_arr, hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label='KalmanNet', ax=axes[0])
        sns.distplot(MSE_KF_linear_arr, hist=False, kde=True, kde_kws={'linewidth': 3}, color='b', label='Kalman Filter', ax=axes[1])
        plt.title("Histogram [Linear]")
        plt.savefig('plt_hist_linear_1')

        ####################
        ### dB Histogram ###
        ####################

        plt.figure(figsize=(50, 20))
        sns.distplot(10 * np.log10(MSE_test_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = 'KalmanNet')
        sns.distplot(10 * np.log10(MSE_KF_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Kalman Filter')
        plt.title("Histogram [dB]")
        plt.savefig('plt_hist_dB')


        fig, axes = plt.subplots(2, 1, figsize=(50, 20), sharey=True, dpi=100)
        sns.distplot(10 * np.log10(MSE_test_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = 'KalmanNet', ax=axes[0])
        sns.distplot(10 * np.log10(MSE_KF_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Kalman Filter', ax=axes[1])
        plt.title("Histogram [dB]")
        plt.savefig('plt_hist_dB_1')

        print('End')


