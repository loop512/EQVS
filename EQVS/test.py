import numpy as np

  
def test():
    dir_em = '/home/cheng/research/steganalysis/steglocator/data/G729CNV_7.5S/embed_matrix2_5/0.1/English/'
    dir_75s = '/home/cheng/research/steganalysis/steglocator/data/G729CNV_7.5S/code_em0.1/English/'
    file_1 = dir_75s + '1.g729a.npy'
    file_2 = dir_75s + '2.g729a.npy'
    file_3 = dir_em + '1.g729a.npy'
    file_4 = dir_em + '2.g729a.npy'
    p1 = np.load(file_1)
    p2 = np.load(file_2)
    p3 = np.load(file_3)
    p4 = np.load(file_4)
    print(p2[0])
    p2 = p2.astype(np.int)
    print(p2[0])


test()
