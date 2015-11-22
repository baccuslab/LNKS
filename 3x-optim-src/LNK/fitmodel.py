#!/usr/bin/env python3
'''
fitmodel.py

This is a script developed for optimizing LNK model.

author: Bongsoo Suh
created: 2015-03-10

(C) 2015 bongsoos
'''

import os
import sys
import cellclass as ccls
import loaddatatools as ldt
import spikingblocks as sb
import optimizationtools as ot
import statstools as stats
import lnkstools as lnks
import numpy as np
import pickle

models = {
    "LNK": lnks.LNK_f,
    }

objectives = {
    "LNK": lnks.LNK_fobj,
    }

bounds = {
    "LNK": lnks.LNK_bnds,
    }


def main():
    '''
    This is the main function for fitting the model.

    regular fitting(each cell)
    all cells fitting
    cross-validation fitting
    '''

    # get environmental variables
    cell_num, model, pathway, objective, crossval, init_num_LNK, num_optims = get_env()

    # fit model
    cell = fit(cell_num, model, np.int(pathway), objective, init_num_LNK, crossval, np.int(num_optims))

    # print results
    print_results(cell)



def fit(cell_num, model, pathway, objective, init_num_LNK, crossval, num_optims):
    '''
    Fit one cell

    Inputs
    ------
    cell_num (string) : cell number
    model (string) : type of model optimized
    pathway (int) : number of pathways for LNK or LNKS model (1 or 2)
    objective (string): type of objective function optimized
    crossval (string): cross-validation 'True' or 'False'
    init_num_LNK (string) : initial parameter of LNK model
    num_optims (int):
        Number of optimization repeated.
        One optimization is MAX_ITER iteration(step gradient).
        Total iteration would be Tot_Iter = MAX_ITER * num_optims
        This way, the optimization process can keep track of intermediate
        cost values, cross-validation(test values) values, and other intermediate
        statistics.
        For each optimization, results are saved

    Optimization
    ------------
        Using the objective function fobj, model function f, and initial parameter theta_init,
        optimize the model and get results, using the method of the cell class.
        cellclass.py module is assumed to be available and in the PYTHONPATH
        optimizationtools.py is assumed to be available and in the PYTHONPATH
        cell.fit: (function, method)
    '''

    # load cell data
    cell = ldt.loadcell(cell_num)

    # select type of model, objective, boundary function
    f = models[model]
    fobj = objectives[objective]
    bnds = bounds[model]
    bound = bnds(pathway=pathway)

    # get initials
    # load LNK initial data
    filename = init_num_LNK + '.initial'
    theta_init = get_initial(filename)


    # save results to these matrices
    thetas = np.zeros([num_optims+1, theta_init.size])
    funs = np.zeros(num_optims+1)
    ccs = np.zeros(num_optims+1)

    # compute initial objective value and correlation coefficient
    thetas[0,:] = theta_init
    funs[0], ccs[0] = compute_init_values(cell, theta_init, model, fobj, f, pathway)

    # Run Optimization
    print("This is in optimize function")
    print("%30s %12s %12s" %("Optimization Process(%)","funs", "corrcoef"))
    print("%30.2f %12.5f %12.5f" %(0, funs[0],ccs[0]))
    for i in range(1, num_optims+1):
        cell.fit(fobj, f, theta_init, model, bound, pathway=pathway)
        thetas[i,:], funs[i], ccs[i] = get_results(cell)
        theta_init = thetas[i,:]

        print("%30.2f %12.5f %12.5f" %( (i/num_optims * 100), funs[i], ccs[i]))

    save_results(cell, cell_num, thetas, funs, ccs)

    return cell

'''
    # cross-validation
    if (crossval == "True") and (not cell_num == "allcells"):
        crossval_fitting(cell_num, model, objective)

    else:
        if cell_num == "allcells":
            allcells_fitting(cell_num, model, objective)
        else:
            regular_fitting(cell_num, model, np.int(pathway), objective, init_num_LNK)
'''

def save_results(cell, cell_num, thetas, funs, ccs):
    '''
    Save optimization results
    '''

    cell.result["thetas"] = thetas
    cell.result["funs"] = funs
    cell.result["corrcoefs"] = ccs
    cell.result["fun_init"] = funs[0]
    cell.result["corrcoef_init"] = ccs[0]

    cell.saveresult(cell_num+'_results.pickle')

    return


def get_results(cell):
    '''
    Get optimization results
    '''
    result = cell.result

    theta = result["theta"]
    fun = result["fun"]
    corrcoef = result["corrcoef"]

    return theta, fun, corrcoef


def compute_init_values(cell, theta_init, model, fobj, f, pathway):
    '''
    Compute the initial objective function value, correlation

    Outputs
    -------
    fun_init (double):
        Initial objective value

    cc_init (double):
        Initial correlation coefficient

    '''
    data = ccls.get_data(cell, model, pathway)

    if pathway:
        y_init = f(theta_init, data[0], pathway=pathway)
        cc_init = stats.corrcoef(y_init, data[1])
        fun_init, grad_init = fobj(theta_init, data[0], data[1], pathway=pathway)

    else:
        y_init = f(theta_init, data[0])
        cc_init = stats.corrcoef(y_init, data[1])
        fun_init, grad_init = fobj(theta_init, data[0], data[1])

    return fun_init, cc_init


def get_env():
    '''
    Assigning input environmental variables to parameters
    '''
    return sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],  sys.argv[5],  sys.argv[6],  sys.argv[7]

def get_initial(filename):
    '''
    Load initial parameter
    '''
    fileObject = open(filename, 'rb')
    initials = pickle.load(fileObject)
    fileObject.close()
    theta_init = initials

    return theta_init


def print_results(cell):
    '''
    Print the final optimization results that will be used to create 3x results

    Outputs
    -------
    fun
    corrcoef
    fun_inits
    corrcoef_inits
    succ
    '''
    fun = np.max(cell.result['fun'])
    if np.isnan(fun) or (fun is None):
        fun = 0
    corrcoef = np.max(cell.result['corrcoef'])
    if np.isnan(corrcoef) or (corrcoef is None):
        corrcoef = 0
    fun_init = np.max(cell.result['fun_init'])
    if np.isnan(fun_init) or (fun_init is None):
        fun_init = 0
    corrcoef_init = np.max(cell.result['corrcoef_init'])
    if np.isnan(corrcoef_init) or (corrcoef_init is None):
        corrcoef_init = 0

    print("cost function value: %12.4f" % fun)
    print("correlation coefficient: %12.4f" % corrcoef)
    print("initial cost function value: %12.4f" % fun_init)
    print("initial correlation coefficient: %12.4f" % corrcoef_init)


'''
command line function call
'''
if __name__ == '__main__':
    main()

