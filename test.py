""" Conduct experiments for TriMine """

import argparse
import os
import shutil
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from trimine import TriMine


if __name__ == '__main__':

    outputdir = '_out/tmp/'
    setseed = False

    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)
    os.makedirs(outputdir)

    if setseed == True:
        np.random.seed(0)

    sns.set()

    # Generate random data
    f1 = 0.5
    f2 = 4
    f3 = 4 # 0.25
    nO = 20
    nA = 20
    T = 50
    b = 40
    # seed1 = b + b * np.sin(np.linspace(-2*np.pi*f1, 2*np.pi*f1, T)) / 2
    # seed2 = b + b * np.sin(np.linspace(-2*np.pi*f2, 2*np.pi*f2, T))
    seed3 = int(b/2) + np.sin(np.linspace(-2*np.pi*f3, 2*np.pi*f3, T))
    seed1 = np.zeros(T)
    seed1[int(T/5):int(T/2)] = b
    seed2 = np.zeros(T)
    seed2[-int(T/4):] = int(b/4)

    plt.plot(seed1, label='topic-1')
    plt.plot(seed2, label='topic-2')
    plt.plot(seed3, label='topic-3')
    plt.xlabel('Time')
    plt.ylabel('Counts')
    plt.legend()
    plt.savefig(outputdir + 'seed.png')
    plt.close()

    topics = np.zeros((nO, nA))
    tensor = np.zeros((nO, nA, T))


    for i in range(nO):
        for j in range(nA):
            # topics[i, j] = z = np.random.randint(3)

            if i < 5:
                topics[i, j] = z = 0
            elif 5 <= i < 10:
                topics[i, j] = z = 1
            else:
                topics[i, j] = z = 2

            if z == 0:
                tensor[i, j, :] = seed1 + np.random.rand(T) * b * .1
            elif z == 1:
                tensor[i, j, :] = seed2 #+ np.random.rand(T) * b * .1
            elif z == 2:
                tensor[i, j, :] = seed3 + np.random.rand(T) * b * .1 

    tensor = np.round(tensor)
    plt.plot(tensor[1,3,:])
    plt.savefig(outputdir + 'example_1.png')
    plt.close()
    plt.plot(tensor[7,2,:])
    plt.savefig(outputdir + 'example_2.png')
    plt.close()
    plt.plot(tensor[12,1,:])
    plt.savefig(outputdir + 'example_3.png')
    plt.close()
    np.save(outputdir + 'tensor.npy', tensor)
    np.savetxt(outputdir + 'seed_topics.txt', topics)

    u, v, n = tensor.shape
    k = 3
    trimine = TriMine(k, u, v, n, outputdir)

    # Infer TriMine's parameters
    start_time = time.process_time()
    trimine.infer(tensor, n_iter=4)
    elapsed_time = time.process_time() - start_time
    print(f'Elapsed time: {elapsed_time:.2f} [sec]')

    trimine.save_model()

    O, A, C = trimine.get_factors()

    plt.plot(O)
    plt.title('Object matrix, O')
    plt.xlabel('Objects')
    plt.ylabel('Topic')
    plt.savefig(outputdir + 'O.png')
    plt.close()

    plt.plot(A)
    plt.title('Actor matrix, A')
    plt.xlabel('Actors')
    plt.ylabel('Topic')
    plt.savefig(outputdir + 'A.png')
    plt.close()

    plt.plot(C)
    plt.title('Time matrix, C')
    plt.xlabel('Time')
    plt.ylabel('Topic')
    plt.savefig(outputdir + 'C.png')
    plt.close()
