#!/usr/bin/env python3
from math import floor
from random import shuffle

def quicksort(x:list[int]) -> list[int]:
    if(not x): return([])
    pivot = floor(len(x)/2)
    smaller = []
    greater = []
    for i in range(len(x)):
        if (i == pivot): pass
        elif (x[i] < x[pivot]):
            smaller.append(x[i])
        else:
            greater.append(x[i])
    return(quicksort(smaller)+x[pivot:pivot+1]+quicksort(greater))
