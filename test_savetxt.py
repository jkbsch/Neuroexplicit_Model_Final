import numpy as np

out_name = ("./Probability_Data/Sleep-EDF-2018-val/")
#out_name = ("./")
np.savetxt(out_name + 'test_data.txt', X=[0, 1, 2, 3, 4], fmt="%d", delimiter=",")
