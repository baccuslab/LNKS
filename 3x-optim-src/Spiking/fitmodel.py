#!/usr/bin/env python3
'''
fitmodel.py

This is a script mainly developed for optimizing Spiking model.

author: Bongsoo Suh
created: 2015-03-10

(C) 2015 bongsoos
'''

import os, sys
import loaddatatools as ldt
import spikingblocks as sb
import optimizationtools as ot
import numpy as np
import pickle
from scipy.stats import pearsonr
import lnkstools as lnks

models = {
        "SCI" : sb.SCI,
        "SC"  : sb.SC,
        "SC1D"  : sb.SC1D,
        "SC1DF"  : sb.SC1DF_C,
        "SC1DF_1"  : sb.SC1DF_C,
        "SCIF": sb.SCIF_C,
        "SCIF2": sb.SCIF2_C,
        "SCIE": sb.SCIE,
        "SCIEF": sb.SCIEF_C,
        "SDF": sb.SDF,
        #"lnk_est": sb.SCIE,
        "lnk_est": sb.SC1DF_C,
        #"lnk_est": sb.SC1D,
        #"LNKS":
        }

objectives = {
        "SCI" : sb.SCI_fobj,
        "SC" : sb.SC_fobj,
        "SC1D"  : sb.SC1D_fobj,
        "SC1DF"  : sb.SC1DF_fobj,
        "SC1DF_1"  : sb.SC1DF_fobj,
        "SCIF": sb.SCIF_fobj,
        "SCIF2": sb.SCIF2_fobj,
        "SCIE": sb.SCIE_fobj,
        "SCIEF": sb.SCIEF_fobj,
        "SDF": sb.SDF_fobj,
        #"lnk_est": sb.SCIE_fobj,
        "lnk_est": sb.SC1DF_fobj,
        #"lnk_est": sb.SC1D_fobj,
        }

bounds = {
        "SCIF": sb.SCIF_bnds,
        "SCIF2": sb.SCIF2_bnds,
        "SCI" :None,
        "SC" : None,
        "SC1D"  : None,
        "SC1DF"  : sb.SC1DF_bnds,
        "SC1DF_1"  : sb.SC1DF_bnds,
        "SCIE" : None,
        "SCIEF": sb.SCIEF_bnds,
        #"lnk_est": None,
        "lnk_est": sb.SC1DF_bnds,
        #"lnk_est": None,
        "SDF": sb.SDF_bnds,
        }


def main():
    '''
    This is the main function for fitting the model.

    regular fitting(each cell)
    all cells fitting
    cross-validation fitting
    '''

    cell_num, model, objective, crossval, gstd = get_env()

    if (crossval == "True") and (not cell_num == "allcells"):
        crossval_fitting(cell_num, model, objective)

    else:
        if cell_num == "allcells":
            allcells_fitting(cell_num, model, objective, np.int(gstd))
        else:
            regular_fitting(cell_num, model, objective, np.int(gstd))


def regular_fitting(cell_num, model, objective, gstd):
    '''
    fit one cell
    '''
    cell = ldt.loadcell(cell_num, gwin_std=gstd)
    norm_range = np.arange(1000,299000)

    if model.lower() == "lnk_est":
        filename = cell_num + "_LNK_results.pickle"
        cell.loadresult(filename)
        theta_LNK = cell.result['theta']
        cell.predict(lnks.LNK, theta_LNK, 'LNK')
        cell.v_est = cell.v_est[norm_range]

    f = models[model]
    fobj = objectives[objective]
    bnds = bounds[model]
    cell.mp = cell.mp[norm_range]
    cell.fr = cell.fr[norm_range]


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

    if model == "SCIF2":
        # initial parameter filename
        if cell_num in ['g4','g12']:
            filename = 'g15.initial'
        else:
            filename = cell_num + '.initial'
        theta_init = np.zeros(8)
        theta_init[:4] = get_initial(filename)
        theta_init[4] = 0
        theta_init[5] = 10 #50,10,80
        theta_init[6] = 0
        theta_init[7] = 500 #500,300,400

    elif model == "SCIE":
        theta_init = np.zeros(10)
        filename = "allcells.initial"
        theta_init = get_initial(filename)

    elif model == "lnk_est":
        # SCIE
        #theta_init = np.zeros(10)
        #filename = "allcells.initial"
        #theta_init = get_initial(filename)

        #filename = cell_num + "_SCIE_results.pickle"
        #cell.loadresult(filename)
        #theta_init = cell.result['theta']

        # SC1DF
        theta_init = np.zeros(7)
        #filename = cell_num + '_SC1DF.initial'
        filename = cell_num + '_SC1DF_lnk.initial'
        init_result = get_initial(filename)
        theta_init[:3] = init_result['theta']
        theta_init[3] = 0
        theta_init[4] = 30 #50,10,80, 10
        theta_init[5] = 0
        theta_init[6] = 500 #500,300,400, 500

        # SC1D
        #theta_init = np.zeros(3)

    elif model == "SC1D":
        theta_init = np.zeros(3)

    elif model == "SC1DF":
        theta_init = np.zeros(7)
        filename = cell_num + '_SC1DF.initial'
        init_result = get_initial(filename)
        theta_init[:3] = init_result['theta']
        theta_init[3] = 0
        theta_init[4] = 80 #50,10,80, 10
        theta_init[5] = 0
        theta_init[6] = 400 #500,300,400, 500

    elif model == "SC1DF_1": # additional feedback
        theta_init = np.zeros(9)
        filename = cell_num + '_SC1DF.initial'
        init_result = get_initial(filename)
        theta_init[:3] = init_result['theta']
        theta_init[3] = 0
        theta_init[4] = 80 #50,10,80, 10
        theta_init[5] = 0
        theta_init[6] = 500 #500,300,400, 500
        theta_init[7] = 0
        theta_init[8] = 10000 #500,300,400, 500

    elif model == "SCIEF":
        #theta_init = np.zeros(12)
        theta_init = np.zeros(14)
        filename = cell_num + "_SCIEF.initial"
        init_result = get_initial(filename)
        theta_init[:10] = init_result['theta']
        #theta_init[11] = 10
        #theta_init[13] = 500
        theta_init[11] = 50
        theta_init[13] = 300

    elif model == "SDF":
        #theta_init = np.zeros(12)
        theta_init = np.zeros(5)
        theta_init[0] = 0.9
        theta_init[2] = 10
        theta_init[4] = 500
    else:
        print(" Error in model name")

    cell = optimize(cell, cell_num, fobj, f, theta_init, model, bnds)
    cell.saveresult(cell_num+'_results.pickle')
    printresults(cell)

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
        corrcoef_crossval[k] = pearsonr(fr_test, fr_est)[0]
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
    return sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

def get_initial(filename):
    '''
    Load initial parameter
    '''
    fileObject = open(filename, 'rb')
    initials = pickle.load(fileObject)
    fileObject.close()
    theta_init = initials

    return theta_init



def optimize(cell, cell_num, fobj, f, theta_init, model, bnds, num_trials=1):
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
    '''

    print("this is in optimize function")
    cell.fit(fobj, f, theta_init, model, bnds, num_trials=num_trials)

    return cell

def printresults(cell):
    fun = np.max(cell.result['fun'])
    if np.isnan(fun) or (fun == None):
        fun = 0
    corrcoef = np.max(cell.result['corrcoef'])
    if np.isnan(corrcoef) or (corrcoef == None):
        corrcoef = 0
    fun_inits = np.max(cell.result['fun_inits'])
    if np.isnan(fun_inits) or (fun_inits == None):
        fun_inits = 0
    corrcoef_inits = np.max(cell.result['corrcoef_inits'])
    if np.isnan(corrcoef_inits) or (corrcoef_inits == None):
        corrcoef_inits = 0

    print("cost function value: %12.4f" % fun)
    print("correlation coefficient: %12.4f" % corrcoef)
    print("initial cost function value: %12.4f" % fun_inits)
    print("initial correlation coefficient: %12.4f" % corrcoef_inits)


'''
command line function call
'''
if __name__ == '__main__':
    main()

