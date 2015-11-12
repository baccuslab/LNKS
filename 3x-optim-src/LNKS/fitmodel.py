#!/usr/bin/env python3
'''
fitmodel.py

This is a script optimizing LNKS model.

author: Bongsoo Suh
created: 2015-03-23
updated: 2015-10-05
updated: 2015-11-12

(C) 2015 bongsoos
'''

import os, sys
import loaddatatools as ldt
import spikingblocks as sb
import optimizationtools as ot
import analysistools as at
import lnkstools as lnks
import numpy as np
import pickle
from scipy.stats import pearsonr

models = {
        "LNKS" : lnks.LNKS_f,
        "LNKS_MP" : lnks.LNKS_MP_f,
        }

models_S = {
        "SCI" : sb.SCI,
        "SCIF" : sb.SCIF,
        "SCIE" : sb.SCIE,
        "SC1DF" : sb.SC1DF_C,
        }

objectives = {
        "LNKS" : lnks.LNKS_fobj,
        "LNKS_MP" : lnks.LNKS_MP_fobj,
        }

bounds = {
        "LNKS": lnks.LNKS_bnds,
        }


def main():
    '''
    This is the main function for fitting the model.

    regular fitting(each cell)
    all cells fitting
    cross-validation fitting
    '''

    cell_num, model, model_S, objective, init_num_LNK, init_num_S, optim_iter, crossval = get_env()

    if (crossval == "True") and (not cell_num == "allcells"):
        crossval_fitting(cell_num, model, objective)

    else:
        if cell_num == "allcells":
            allcells_fitting(cell_num, model, objective)
        else:
            if np.int(optim_iter) == 0:
                regular_fitting(cell_num, model, model_S, objective, init_num_LNK, init_num_S)
            else:
                regular_fitting_iter(cell_num, model, model_S, objective, init_num_LNK, init_num_S, optim_iter)


def regular_fitting(cell_num, model, model_S, objective, init_num_LNK, init_num_S):
    '''
    fit one cell
    '''
    cell = ldt.loadcell(cell_num)
    f = models[model]
    fS = models_S[model_S]
    fobj = objectives[objective]
    bnds = bounds[model]


    # get initials
    theta_init = np.zeros(17 + 10)

    filename = init_num_LNK + '.initial'
    theta_LNK = get_initial(filename)
    filename = init_num_S + '.initial'
    theta_S = get_initial(filename)

    theta_init[:17] = theta_LNK['theta']
    theta_init[17:] = theta_S

    cell = optimize(cell, cell_num, fobj, f, theta_init, model, bnds)
    theta = cell.result['theta']
    cell.predict(f, theta, model)
    mp_range = np.arange(1000, 300000)
    cell.result['corrcoef_mp'] = at.corrcoef(cell.v_est[mp_range], cell.mp[mp_range])
    cell.saveresult(cell_num+'_results.pickle')
    printresults(cell)

    print("optimization finished")


def regular_fitting_iter(cell_num, model, model_S, objective, init_num_LNK, init_num_S, optim_iter):
    '''
    fit one cell with iteration
    '''
    cell = ldt.loadcell(cell_num)
    f = models[model]
    fS = models_S[model_S]
    fobj = objectives[objective]
    bnds = bounds[model]

    # get initials
    theta_init = np.zeros(17 + 10)

    filename = init_num_LNK + '.initial'
    theta_LNK = get_initial(filename)
    filename = init_num_S + '.initial'
    theta_S = get_initial(filename)

    theta_init[:17] = theta_LNK['theta']
    theta_init[17:] = theta_S

    cell.result_iter = []
    for i in range(np.int(optim_iter)):
        '''
        bnd_mode (int):
            Different modes of LNKS model parameter boundary for optimization
            0: fit LNKS model
            1: fit LNK (S fixed)
            2: fit S (LNK fixed)
        '''
        bnd_modes = [1, 2]
        for bnd_mode in bnd_modes:
            cell = optimize(cell, cell_num, fobj, f, theta_init, model, bnds, bnd_mode)
            theta = cell.result['theta']
            cell.predict(f, theta, model)
            mp_range = np.arange(1000, 300000)
            cell.result['corrcoef_mp'] = at.corrcoef(cell.v_est[mp_range], cell.mp[mp_range])
            theta_init = theta

        '''
        save cell result into list as a tuple (iter number, bound mode, result)
        '''
        cell.result_iter.append((i, bnd_mode, cell.result))

    bnd_mode = 0
    cell = optimize(cell, cell_num, fobj, f, theta_init, model, bnds, bnd_mode)
    theta = cell.result['theta']
    cell.predict(f, theta, model)
    mp_range = np.arange(1000, 300000)
    cell.result['corrcoef_mp'] = at.corrcoef(cell.v_est[mp_range], cell.mp[mp_range])

    '''
    save cell result into list as a tuple (iter number, bound mode, result)
    '''
    cell.result_iter.append((i, bnd_mode, cell.result))
    printresults(cell)

    cell.result = cell.result_iter
    cell.saveresult(cell_num+'_results.pickle')

    print("optimization finished")


def crossval_fitting(cell_num, model, objective):
    '''
    fit each cell using the training data of [1s, 10s, 20s, 50s, 100s, 150s, 200s, 240s]
    and test on the test data.
    '''
    cell = ldt.loadcell(cell_num)
    f = models[model]
    fobj = objectives[objective]
    bnds = bounds[model]

    if model.lower() == "scif":
        theta_size = 6
    elif model.lower() == "scie":
        theta_size = 10
    else:
        theta_size = 4
    train_data = np.array([1, 10, 20, 50, 100, 150, 200, 240])
    corrcoef_crossval = np.zeros(train_data.shape)
    theta_crossval = np.zeros((train_data.size, theta_size))

    k = 0
    mp_true = cell.mp
    fr_true = cell.fr
    norm_range = np.arange(1000, 299000)
    mp_true = mp_true - np.min(mp_true[norm_range])
    mp_true = mp_true/np.max(mp_true[norm_range])
    fr_true = fr_true/np.max(fr_true[norm_range])
    for i in train_data:
        train_range = np.arange(1000, (i+1)*1000)
        test_range = np.arange((i+1)*1000, 300000)

        mp_test = mp_true[test_range]
        fr_test = fr_true[test_range]

        cell.mp = mp_true[train_range]
        cell.fr = fr_true[train_range]

        # get initials
        if model == "SCIF":
            # initial parameter filename
            if cell_num in ['g4','g12']:
                filename = 'g15.initial'
            else:
                filename = cell_num + '.initial'
            theta_init = np.zeros(6)
            theta_init[:4] = get_initial(filename)
            theta_init[4] = 0
            theta_init[5] = 100

        elif model == "SCIE":
            theta_init = np.zeros(10)
            filename = "allcells.initial"
            theta_init = get_initial(filename)

        else:
            print(" Error in model name")

        cell = optimize(cell, cell_num, fobj, f, theta_init, model, bnds)

        fr_est = f(cell.result['theta'], mp_test)
        #corrcoef_crossval[k] = pearsonr(fr_test, fr_est)[0]
        corrcoef_crossval[k] = at.corrcoef(fr_test, fr_est)
        theta_crossval[k,:] = cell.result['theta']

        k += 1

    cell.result['corrcoef_crossval'] = corrcoef_crossval
    cell.result['theta_crossval'] = theta_crossval
    cell.saveresult(cell_num+'_results.pickle')
    printresults(cell)

    print("optimization finished")


def allcells_fitting(cell_num, model, objective):
    '''
    fit all cells together to get the average spiking block(the common spiking block) that captures
    all the spikings in all the cells as much as possible
    '''
    cells = ldt.loadcells()

    norm_range = np.arange(1000,299000)
    #N = cells['g9'].st.size
    N = norm_range.size
    keys = [key for key in cells.keys() if not key in ['g4','g12']]

    mp = np.zeros(N * len(keys))
    fr = np.zeros(N * len(keys))

    i = 0
    for key in keys:
        temp_mp = cells[key].mp[norm_range]
        temp_mp = temp_mp - np.min(temp_mp)
        temp_mp = temp_mp / np.max(temp_mp)

        temp_fr = cells[key].fr[norm_range]
        temp_fr = temp_fr / np.max(temp_fr)

        mp[i*N:(i+1)*N] = temp_mp
        fr[i*N:(i+1)*N] = temp_fr

        i += 1

    cell = ldt.loadcell('g9')

    cell.mp = mp
    cell.fr = fr

    f = models[model]
    fobj = objectives[objective]
    bnds = bounds[model]

    # initial parameter filename
    filename = 'g17.initial'

    # get initials
    if model == "SCIF":
        theta_init = np.zeros(6)
        theta_init[:4] = get_initial(filename)
        theta_init[4] = 0
        theta_init[5] = 100

    elif model == "SCIE":
        theta_init = np.zeros(10)
        theta_init[:4] = get_initial(filename)

    else:
        print(" Error in model name")

    cell = optimize(cell, cell_num, fobj, f, theta_init, model, bnds)
    cell.saveresult(cell_num+'_results.pickle')
    printresults(cell)

    print("optimization finished")


def get_env():
    '''
    Assigning input environmental variables to parameters
    '''
    return sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8]

def get_initial(filename):
    '''
    Load initial parameter
    '''
    fileObject = open(filename, 'rb')
    initials = pickle.load(fileObject)
    fileObject.close()
    theta_init = initials

    return theta_init



def optimize(cell, cell_num, fobj, f, theta_init, model, bnds, bnd_mode=0):
    '''
    * Optimization *
        Using the objective function fobj, model function f, and initial parameter theta_init,
        optimize the model and get results, using the method of the cell class.
        cellclass.py module is assumed to be available and in the PYTHONPATH
        optimizationtools.py is assumed to be available and in the PYTHONPATH

        cell.fit: (function, method)
            Inputs
            ------
                f(theta, x_in)
                fobj(theta, x_in, y)

    Inputs
    ------
    bnd_mode (int):
        Different modes of LNKS model parameter boundary for optimization
        0: fit LNKS model
        1: fit LNK (S fixed)
        2: fit S (LNK fixed)
    '''

    print("this is in optimize function")

    bound = bnds(bnd_mode)

    cell.fit(fobj, f, theta_init, model, bound)

    return cell

def printresults(cell):
    fun = np.max(cell.result['fun'])
    if np.isnan(fun) or (fun == None):
        fun = 0

    corrcoef = np.max(cell.result['corrcoef'])
    if np.isnan(corrcoef) or (corrcoef == None):
        corrcoef = 0

    corrcoef_mp = np.max(cell.result['corrcoef_mp'])
    if np.isnan(corrcoef_mp) or (corrcoef_mp == None):
        corrcoef_mp = 0

    fun_inits = np.max(cell.result['fun_inits'])
    if np.isnan(fun_inits) or (fun_inits == None):
        fun_inits = 0

    corrcoef_inits = np.max(cell.result['corrcoef_inits'])
    if np.isnan(corrcoef_inits) or (corrcoef_inits == None):
        corrcoef_inits = 0

    if cell.result['success'] == True:
        succ = 1
    elif cell.result['success'] == False:
        succ = 0

    print("cost function value: %12.4f" % fun)
    print("correlation coefficient: %12.4f" % corrcoef)
    print("correlation coefficient mp: %12.4f" % corrcoef_mp)
    print("initial cost function value: %12.4f" % fun_inits)
    print("initial correlation coefficient: %12.4f" % corrcoef_inits)
    print("optimization output status: %d" %succ)


'''
command line function call
'''
if __name__ == '__main__':
    main()

