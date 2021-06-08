# -*- coding: utf-8 -*-

import cv2
import numpy
import numpy as np
from matplotlib import pyplot as plt
import math
import copy
import sys
import os
from time import sleep
from threading import Thread
from multiprocessing import Process, Lock, Queue
import re

np.set_printoptions(threshold=numpy.nan)

class Kappa:
    def __init__(self, TP, FN, FP, TN):
        self._tableau_contingence = np.array([[TP, FP], [FN, TN]], dtype=numpy.float64) # matrice de confusion
        
        self._p = np.array([TP+FP, FN+TN], dtype=numpy.float64) # totaux pour predit
        self._r = np.array([TP+FN, FP+TN], dtype=numpy.float64) # totaux pour réel
        
        self._n = sum(sum(self._tableau_contingence)) # total du tableau
    
    # calcule la proposition de l'accord observé
    def po(self):
        tc = self._tableau_contingence
        # la somme de la diagonal / la somme de la matrice
        return sum(float(tc[i][i]) for i in range(len(tc))) / self._n

    # calcule la proposition de l'accord aleatoire (valeur espéré)
    def pe(self):
        tc = self._tableau_contingence
        # produit cartesien de p et r / la somme de la matrice au carré
        return sum(float(self._p[i] * self._r[i]) for i in range(len(tc))) / ( self._n ** 2 )

    # calcule le coef de kappa
    def kappa(self):
        po = self.po()
        pe = self.pe()
        return (po - pe) / (1 - pe)

def main():
    file = open("XXX.txt", 'r')
    wfile = open("YYY.txt", 'w')
    lines = [re.findall(r'\d+', i) for i in file.readlines()]

    print '0> ', lines

    sum_po = 0
    sum_pe = 0
    sum_kappa = 0
    count = 0
    
    wfile.write('{}\t\t{}\t\t{}\n'.format('Po','Pe','Kappa'))
    wfile.write('------------------------\n')

    for line in lines:
        print '0> ', line
        TP = int(line[0])
        FN = int(line[1])
        FP = int(line[2])
        TN = int(line[3])

        print '1> ', TP, FN, FP, TN

        k = Kappa(TP, FN, FP, TN)
        po = round(k.po(), 3)
        print '2> ', po
        pe = round(k.pe(), 3)
        print '3> ', pe
        kappa = round(k.kappa(), 3)
        print '4> ', kappa

        sum_po += po
        sum_pe += pe
        sum_kappa += kappa
        count += 1

        wfile.write('{}\t{}\t{}\n'.format(po, pe, kappa))

    sum_po = round(float(sum_po) / count, 3)
    sum_pe = round(float(sum_pe) / count, 3)
    sum_kappa = round(float(sum_kappa) / count, 3)

    print '5> ', sum_po
    print '6> ', sum_pe
    print '7> ', sum_kappa
    wfile.write('------------------------\n')
    wfile.write('{}\t{}\t{}\n'.format(sum_po, sum_pe, sum_kappa))

if __name__ == '__main__':
    main()
