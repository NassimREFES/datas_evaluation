# -*- coding: utf-8 -*-

from __future__ import print_function
import unicodedata
import platform
import subprocess

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

np.set_printoptions(threshold=numpy.nan)

# --------------------------------------------- #

# white-color to white-black
# pour les image manuel
def from_wc_to_wb(img):
    rows = img.shape[0]
    cols = img.shape[1]

    if len(img.shape) == 3L:
        b, g, r = cv2.split(img)
        for row in range(rows):
            for col in range(cols):
                if b[row][col] == 255 and g[row][col] == 255 and b[row][col] == 255:
                    b[row][col] = g[row][col] = r[row][col] = 255
                    pass
                else:
                    b[row][col] = g[row][col] = r[row][col] = 0
        return cv2.merge((b, g, r))
    return img

def matrice_confusion(img_source, img_compare):
    mc = {
        'TP' : 0,
        'FN' : 0,
        'FP' : 0,
        'TN' : 0,
    }

    rows = img_compare.shape[0]
    cols = img_compare.shape[1]

    ombre_val = 255

    if len(img_source.shape) == 3L:
        Bs, Gs, Rs = cv2.split(img_source)
        Bc, Gc, Rc = cv2.split(img_compare)

        for row in range(rows):
            for col in range(cols):
                src_ombre = (Bs[row][col] == ombre_val and Gs[row][col] == ombre_val and Rs[row][col] == ombre_val)
                src_non_ombre = (Bs[row][col] != ombre_val and Gs[row][col] != ombre_val and Rs[row][col] != ombre_val)
                comp_ombre = (Bc[row][col] == ombre_val and Gc[row][col] == ombre_val and Rc[row][col] == ombre_val)
                comp_non_ombre = (Bc[row][col] != ombre_val and Gc[row][col] != ombre_val and Rc[row][col] != ombre_val)

                if src_ombre and comp_ombre:
                    mc['TP'] += 1
                elif src_ombre and comp_non_ombre:
                    mc['FN'] += 1
                elif src_non_ombre and comp_ombre:
                    mc['FP'] += 1
                elif src_non_ombre and comp_non_ombre:
                    mc['TN'] += 1
                else:
                    pass
    return mc

def precision_producteur(mc):
    if (mc['TP'] + mc['FN']) == 0:
        ns = 0
    else:
        ns = float(mc['TP']) / (mc['TP'] + mc['FN']) # sensibilite
    if (mc['FP'] + mc['TN']) == 0:
        nn = 0
    else:
        nn = float(mc['TN']) / (mc['FP'] + mc['TN']) # specificite

    return ns, nn

def precision_utilisateur(mc):
    if (mc['TP'] + mc['FP']) == 0:
        ps = 0
    else:
        ps = float(mc['TP']) / (mc['TP'] + mc['FP']) # precision
    if (mc['TN'] + mc['FN']) == 0:
        pn = 0
    else:
        pn = float(mc['TN']) / (mc['TN'] + mc['FN']) # valeur predictive negative

    return ps, pn

def precision_globale(mc):
    return float((mc['TP'] + mc['TN'])) / (mc['TP'] + mc['TN'] + mc['FP'] + mc['FN'])

def main():
    global g_seeds
    if len(sys.argv) >= 3:
        img_manuel = cv2.imread(sys.argv[1])

        img_algo = cv2.imread(sys.argv[2])

        filename =  unicode(sys.argv[3]) # nom du fichier ou ajouté le resultat

        imagename =  unicode(sys.argv[4]) # nom de l'image

        res = from_wc_to_wb(img_manuel)

        # ------------------------------------------------------------------------------------------------
        # evaluation

        mc = matrice_confusion(res, img_algo)
        pp = precision_producteur(mc)
        pu = precision_utilisateur(mc)
        pg = precision_globale(mc)

        pp = ( pp[0] * 100, pp[1] * 100 )
        pu = ( pu[0] * 100, pu[1] * 100 )
        pg *= 100

        print ('TP= ', mc['TP'], ' FN= ', mc['FN'], ' FP= ', mc['FP'], ' TN= ', mc['TN'])
        print ('Pp= (ns:', round(pp[0], 3) , ', nn:', round(pp[1], 3), ')')
        print ('Pu= (ps:', round(pu[0], 3) , ', pn:', round(pu[1], 3), ')')
        print ('Pg= ', round(pg, 3))

        with open(filename, "a") as xfile:
            xfile.write("{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\n".format(imagename, mc['TP'], mc['FN'], mc['FP'], mc['TN'], 
                                                                    round(pp[0], 3), round(pp[1], 3), round(pu[0], 3), round(pu[1], 3),
                                                                    round(pg, 3)))

        # ------------------------------------------------------------------------------------------------
        
        # while True:
        #     cv2.imshow('img', img_manuel)
        #     cv2.imshow('res', res)
        #     #cv2.imshow('ferm', fermeture_morph)
        #     k = cv2.waitKey(33)
        #     if k==27:
        #         break
        #     elif k==-1:
        #         continue
        #     else:
        #         print (k)

if __name__ == '__main__':
    main()
