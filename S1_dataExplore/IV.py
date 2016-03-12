#coding = 'utf-8'
"""data explore for raw datasheet"""
__author__ = "changandao&jiangweiwu"
__date__   = "2016.3.8"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def WOEFunc(Aframe, k):
    newFrame = Aframe.set_index('Idx')
    goodN = newFrame.groupby('target').get_group(0)
    badN = newFrame.groupby('target').get_group(1)
    for clm in newFrame.columns:
        Maxmum = newFrame[clm].max()
        Minmum = newFrame[clm].min()
        distance = (Maxmum - Minmum)/k
