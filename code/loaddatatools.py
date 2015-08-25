#!/usr/bin/env python3
'''
loaddatatools.py

Tools for loading cell data

cell names
    'g4', 'g6', 'g8', 'g9', 'g9apb', 'g11', 'g12', 'g13', 'g15', 'g17', 'g17apb', 'g19', 'g19apb' ,'g20', 'g20aob'

METHODS
    loadcells
    ---------
        load all cell data into a dictionary
        keys are the cell names ('g4', 'g6', 'g8', ... )
        usage:
            cells = loadcells()

    loadcell
    --------
        load one cell data into a class object
        usage:
            g11 = loadcell('g11')


author: Bongsoo Suh

'''

import cellclass
import sys
import platform
import socket

'''
setting custom path
    path:
        the path to the directory where the data files are located.
'''
hostname = socket.gethostname()
if 'corn' in hostname.lower():
    path = '~/lnks/data/'
elif 'bongsoo' in hostname.lower():
    path = '/Users/suh/Projects/expts/expt00/'
elif 'sni-vcs-baccus' in hostname.lower():
    path = '~/Projects/expts/expt00/'
else:
    message = "Hostname: " + hostname + " is not known. The data path needs to be specified."
    sys.exit(message)

def loadcells(gwin_std=10):
    '''
    load all cells and returns cells structure(or dictionary)
    '''
    cells = { 'g4' : g4(gwin_std=gwin_std),
            'g6' : g6(gwin_std=gwin_std),
            'g8' : g8(gwin_std=gwin_std),
            'g9' : g9(gwin_std=gwin_std),
            'g9apb' : g9apb(gwin_std=gwin_std),
            'g11' : g11(gwin_std=gwin_std),
            'g12' : g12(gwin_std=gwin_std),
            'g13' : g13(gwin_std=gwin_std),
            'g15' : g15(gwin_std=gwin_std),
            'g17' : g17(gwin_std=gwin_std),
            'g17apb' : g17apb(gwin_std=gwin_std),
            'g19' : g19(gwin_std=gwin_std),
            'g19apb' : g19apb(gwin_std=gwin_std),
            'g20' : g20(gwin_std=gwin_std),
            'g20apb' : g20apb(gwin_std=gwin_std),
            }
    return cells


def loadcell(cell_num, gwin_std=10):
    '''
    load cell given cell number(string)
    cell numbers (string):
        g4, g6, g11, g20 ,,,
    '''
    if cell_num == 'g4':
        cell = g4(gwin_std=gwin_std)

    elif cell_num == 'g6':
        cell = g6(gwin_std=gwin_std)

    elif cell_num == 'g8':
        cell = g8(gwin_std=gwin_std)

    elif cell_num == 'g9':
        cell = g9(gwin_std=gwin_std)

    elif cell_num == 'g9apb':
        cell = g9apb(gwin_std=gwin_std)

    elif cell_num == 'g11':
        cell = g11(gwin_std=gwin_std)

    elif cell_num == 'g12':
        cell = g12(gwin_std=gwin_std)

    elif cell_num == 'g13':
        cell = g13(gwin_std=gwin_std)

    elif cell_num == 'g15':
        cell = g15(gwin_std=gwin_std)

    elif cell_num == 'g17':
        cell = g17(gwin_std=gwin_std)

    elif cell_num == 'g17apb':
        cell = g17apb(gwin_std=gwin_std)

    elif cell_num == 'g19':
        cell = g19(gwin_std=gwin_std)

    elif cell_num == 'g19apb':
        cell = g19apb(gwin_std=gwin_std)
    elif cell_num == 'g20':
        cell = g20(gwin_std=gwin_std)

    elif cell_num == 'g20apb':
        cell = g20apb(gwin_std=gwin_std)

    else:
        print('cell number error')

    return cell

def g4(fc=3, threshold=5, filt_len=10, num_rep=5, gwin_len=1000, gwin_std=10):
    filename = 'g4.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': False}

    gcell = cellclass.Cell()
    gcell.loadcell(options)

    return gcell

def g6(fc=3, threshold=3, filt_len=10, num_rep=5, gwin_len=1000, gwin_std=10):
    filename = 'g6.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': False}

    gcell = cellclass.Cell()
    gcell.loadcell(options)

    return gcell

def g8(fc=3, threshold=3, filt_len=10, num_rep=2, gwin_len=1000, gwin_std=10):
    filename = 'g8.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': False}

    gcell = cellclass.Cell()
    gcell.loadcell(options)

    return gcell

def g9(fc=3, threshold=5, filt_len=10, num_rep=2, gwin_len=1000, gwin_std=10):
    filename = 'g9.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': False}

    gcell = cellclass.Cell()
    gcell.loadcell(options)

    return gcell

def g9apb(fc=3, threshold=5, filt_len=10, num_rep=2, gwin_len=1000, gwin_std=10):
    filename = 'g9.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': True}

    gcell = cellclass.Cell()
    gcell.loadcell(options)

    return gcell

def g11(fc=3, threshold=1, filt_len=10, num_rep=10, gwin_len=1000, gwin_std=10):
    filename = 'g11.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': False}

    gcell = cellclass.Cell()
    gcell.loadcell(options)

    return gcell

def g12(fc=10, threshold=5, filt_len=10, num_rep=5, gwin_len=1000, gwin_std=10):
    filename = 'g12.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': False}

    gcell = cellclass.Cell()
    gcell.loadcell(options)

    return gcell

def g13(fc=12, threshold=3, filt_len=10, num_rep=8, gwin_len=1000, gwin_std=10):
    filename = 'g13.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': False}

    gcell = cellclass.Cell()
    gcell.loadcell(options)

    return gcell

def g15(fc=3, threshold=3, filt_len=10, num_rep=5, gwin_len=1000, gwin_std=10):
    filename = 'g15.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': False}

    gcell = cellclass.Cell()
    gcell.loadcell(options)

    return gcell

def g17(fc=3, threshold=5, filt_len=10, num_rep=5, gwin_len=1000, gwin_std=10):
    filename = 'g17.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': False}

    gcell = cellclass.Cell()
    gcell.loadcell(options)

    return gcell

def g17apb(fc=3, threshold=5, filt_len=10, num_rep=5, gwin_len=1000, gwin_std=10):
    filename = 'g17.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': True}

    gcell = cellclass.Cell()
    gcell.loadcell(options)

    return gcell

def g19(fc=3, threshold=5, filt_len=10, num_rep=5, gwin_len=1000, gwin_std=10):
    filename = 'g19.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': False}

    gcell = cellclass.Cell()
    gcell.loadcell(options)

    return gcell

def g19apb(fc=3, threshold=5, filt_len=10, num_rep=5, gwin_len=1000, gwin_std=10):
    filename = 'g19.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': True}

    gcell = cellclass.Cell()
    gcell.loadcell(options)

    return gcell

def g20(fc=3, threshold=5, filt_len=10, num_rep=5, gwin_len=1000, gwin_std=10):
    filename = 'g20.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': False}

    gcell = cellclass.Cell()
    gcell.loadcell(options)

    return gcell

def g20apb(fc=3, threshold=5, filt_len=10, num_rep=5, gwin_len=1000, gwin_std=10):
    filename = 'g20.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': True}

    gcell = cellclass.Cell()
    gcell.loadcell(options)

    return gcell

