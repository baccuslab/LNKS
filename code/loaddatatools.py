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

import cellclass as _ccls
import sys as _sys
import platform
import socket as _socket
import multiprocessing as _mult

'''
setting custom path
    path:
        the path to the directory where the data files are located.
'''
hostname = _socket.gethostname()
if 'corn' in hostname.lower():
    path = '/farmshare/user_data/bssuh/LNKS/data/'
elif 'bongsoo' in hostname.lower():
    path = '/Users/suh/Projects/expts/expt00/'
elif 'sni-vcs-baccus' in hostname.lower():
    path = '/home/bssuh/Projects/expts/expt00/'
else:
    message = "Hostname: " + hostname + " is not known. The data path needs to be specified."
    _sys.exit(message)

def loadcells(gwin_std=10):
    '''
    load all cells and returns cells structure(or dictionary)
    '''
    pool = _mult.Pool(processes=_mult.cpu_count())
    cell_nums = ['g4', 'g6', 'g8', 'g9', 'g9apb', 'g11', 'g12', 'g13', 'g15', 'g17', 'g17apb', 'g19', 'g19apb', 'g20', 'g20apb']
    results = [(cell_num, pool.apply_async(loadcell, args=(cell_num, gwin_std))) for cell_num in cell_nums]
    pool.close()
    pool.join()
    return {cell_num:p.get() for cell_num, p in results}


def loadcell(cell_num, gwin_std=10):
    '''
    load cell given cell number(string)
    cell numbers (string):
        g4, g6, g11, g20 ,,,
    '''
    if cell_num == 'g4':
        cell = _g4(gwin_std=gwin_std)

    elif cell_num == 'g6':
        cell = _g6(gwin_std=gwin_std)

    elif cell_num == 'g8':
        cell = _g8(gwin_std=gwin_std)

    elif cell_num == 'g9':
        cell = _g9(gwin_std=gwin_std)

    elif cell_num == 'g9apb':
        cell = _g9apb(gwin_std=gwin_std)

    elif cell_num == 'g11':
        cell = _g11(gwin_std=gwin_std)

    elif cell_num == 'g12':
        cell = _g12(gwin_std=gwin_std)

    elif cell_num == 'g13':
        cell = _g13(gwin_std=gwin_std)

    elif cell_num == 'g15':
        cell = _g15(gwin_std=gwin_std)

    elif cell_num == 'g17':
        cell = _g17(gwin_std=gwin_std)

    elif cell_num == 'g17apb':
        cell = _g17apb(gwin_std=gwin_std)

    elif cell_num == 'g19':
        cell = _g19(gwin_std=gwin_std)

    elif cell_num == 'g19apb':
        cell = _g19apb(gwin_std=gwin_std)

    elif cell_num == 'g20':
        cell = _g20(gwin_std=gwin_std)

    elif cell_num == 'g20apb':
        cell = _g20apb(gwin_std=gwin_std)

    else:
        print('cell number error')

    return cell

def _g4(fc=3, threshold=5, filt_len=10, num_rep=5, gwin_len=1000, gwin_std=10):
    filename = 'g4.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': False}

    gcell = _ccls.Cell()
    gcell.loadcell(options)

    return gcell

def _g6(fc=3, threshold=3, filt_len=10, num_rep=5, gwin_len=1000, gwin_std=10):
    filename = 'g6.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': False}

    gcell = _ccls.Cell()
    gcell.loadcell(options)

    return gcell

def _g8(fc=3, threshold=3, filt_len=10, num_rep=2, gwin_len=1000, gwin_std=10):
    filename = 'g8.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': False}

    gcell = _ccls.Cell()
    gcell.loadcell(options)

    return gcell

def _g9(fc=3, threshold=5, filt_len=10, num_rep=2, gwin_len=1000, gwin_std=10):
    filename = 'g9.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': False}

    gcell = _ccls.Cell()
    gcell.loadcell(options)

    return gcell

def _g9apb(fc=3, threshold=5, filt_len=10, num_rep=2, gwin_len=1000, gwin_std=10):
    filename = 'g9.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': True}

    gcell = _ccls.Cell()
    gcell.loadcell(options)

    return gcell

def _g11(fc=3, threshold=1, filt_len=10, num_rep=10, gwin_len=1000, gwin_std=10):
    filename = 'g11.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': False}

    gcell = _ccls.Cell()
    gcell.loadcell(options)

    return gcell

def _g12(fc=10, threshold=5, filt_len=10, num_rep=5, gwin_len=1000, gwin_std=10):
    filename = 'g12.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': False}

    gcell = _ccls.Cell()
    gcell.loadcell(options)

    return gcell

def _g13(fc=12, threshold=3, filt_len=10, num_rep=8, gwin_len=1000, gwin_std=10):
    filename = 'g13.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': False}

    gcell = _ccls.Cell()
    gcell.loadcell(options)

    return gcell

def _g15(fc=3, threshold=3, filt_len=10, num_rep=5, gwin_len=1000, gwin_std=10):
    filename = 'g15.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': False}

    gcell = _ccls.Cell()
    gcell.loadcell(options)

    return gcell

def _g17(fc=3, threshold=5, filt_len=10, num_rep=5, gwin_len=1000, gwin_std=10):
    filename = 'g17.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': False}

    gcell = _ccls.Cell()
    gcell.loadcell(options)

    return gcell

def _g17apb(fc=3, threshold=5, filt_len=10, num_rep=5, gwin_len=1000, gwin_std=10):
    filename = 'g17.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': True}

    gcell = _ccls.Cell()
    gcell.loadcell(options)

    return gcell

def _g19(fc=3, threshold=5, filt_len=10, num_rep=5, gwin_len=1000, gwin_std=10):
    filename = 'g19.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': False}

    gcell = _ccls.Cell()
    gcell.loadcell(options)

    return gcell

def _g19apb(fc=3, threshold=5, filt_len=10, num_rep=5, gwin_len=1000, gwin_std=10):
    filename = 'g19.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': True}

    gcell = _ccls.Cell()
    gcell.loadcell(options)

    return gcell

def _g20(fc=3, threshold=5, filt_len=10, num_rep=5, gwin_len=1000, gwin_std=10):
    filename = 'g20.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': False}

    gcell = _ccls.Cell()
    gcell.loadcell(options)

    return gcell

def _g20apb(fc=3, threshold=5, filt_len=10, num_rep=5, gwin_len=1000, gwin_std=10):
    filename = 'g20.mat'
    options = {'filename': path+filename, 'fc': fc, 'threshold': threshold, 'filt_len': filt_len, 'num_rep': num_rep, 'gwin_len': gwin_len, 'gwin_std': gwin_std, 'apbflag': True}

    gcell = _ccls.Cell()
    gcell.loadcell(options)

    return gcell

