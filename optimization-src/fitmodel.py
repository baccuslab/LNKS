#!/usr/bin/env python3
'''
fitmodel.py

This is a script developed for optimizing LNK model.

author: Bongsoo Suh
created: 2015-03-10
updated: 2015-11-20

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
import pdb
import time
import pandas as pd

models = {
    "LNK": lnks.LNK_f,
    "LNKS": lnks.LNKS_f,
    "LNKS_MP": lnks.LNKS_MP_f,
    "Spiking": sb.SC1DF_C,
    }

objectives = {
    "LNK": lnks.LNK_fobj,
    "LNKS": lnks.LNKS_fobj,
    "LNKS_MP": lnks.LNKS_MP_fobj,
    "Spiking": sb.SC1DF_fobj,
    }

bounds = {
    "LNK": lnks.LNK_bnds,
    "LNKS": lnks.LNKS_bnds,
    "LNKS_MP": lnks.LNKS_bnds,
    "Spiking": sb.SC1DF_bnds,
    }

# Set Initial Path
PATH_INITIAL_LNK = './initial_LNK/'
PATH_INITIAL_S = './initial_S/'

def main():
    '''
    This is the main function for fitting the model.

    regular fitting(each cell)
    all cells fitting
    cross-validation fitting
    '''

    # get environmental variables
    cell_num, model, objective, pathway, init_num_LNK, init_num_S, num_optims, crossval, is_grad = get_env()
    init_num = (init_num_LNK, init_num_S)

    options = {}
    options['pathway'] = np.int(pathway)
    options['model'] = model

    if crossval == "True":
        options['crossval'] = True
    else:
        options['crossval'] = False

    if is_grad == "True":
        options['is_grad'] = True
    else:
        options['is_grad'] = False

    # fit model
    cell = fit(cell_num, model, objective, init_num, np.int(num_optims), options)

    # print results
    print_results(cell, cell_num, model, objective, pathway, init_num_LNK, init_num_S, num_optims, crossval, is_grad)



def fit(cell_num, model, objective, init_num, num_optims, options):
    '''
    Fit one cell

    Inputs
    ------
    cell_num (string) : cell number
    model (string) : type of model optimized
    objective (string): type of objective function optimized
    init_num (string) : initial parameter of model
    num_optims (int):
        Number of optimization repeated.
        One optimization is MAX_ITER iteration(step gradient).
        Total iteration would be Tot_Iter = MAX_ITER * num_optims
        This way, the optimization process can keep track of intermediate
        cost values, cross-validation(test values) values, and other intermediate
        statistics.
        For each optimization, results are saved
    options (dictionary)
        pathway (int) : number of pathways for LNK or LNKS model (1 or 2)
        crossval (bool): cross-validation 'True' or 'False'
        is_grad (bool): gradient On(True) or Off(False)

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
    bound = bnds(pathway=options['pathway'])

    # get initials
    theta_init = get_initial(model, init_num)

    # compute initial objective value and correlation coefficient
    fun_train, cc_train, evar_train = compute_func_values(cell,theta_init,model,fobj, f,options, True)
    fun_test, cc_test, evar_test = compute_func_values(cell,theta_init,model, fobj,f, options, False)

    # Print if test mode
    if num_optims < 100:
        # Run Optimization
        print("\nFit cell %s. Optimize %s model using %s objective function\n" %(cell_num, model, objective))
        print("%30s %17s %17s %17s %17s %17s %17s" %("Optimization Process(%)","Update Time(sec)","funs",
                                                "corrcoef(train)","var-expl(train)",
                                                "corrcoef(test)", "var-expl(test)"))
        print("%30.2f %17.2f %17.5f %17.5f %17.5f %17.5f %17.5f" %(0, 0, fun_train,cc_train,evar_train,cc_test,evar_test))

    # save results to Data Frame
    idxs = np.array(range(num_optims+1))
    cols = ["Optimization Process(%)","Update Time(sec)","funs","corrcoef(train)","var-expl(train)","corrcoef(test)","var-expl(test)"]
    df = pd.DataFrame(index=idxs, columns=cols)
    df.loc[0] = (0, 0, fun_train,cc_train,evar_train,cc_test,evar_test)

    df_thetas = pd.DataFrame(index=idxs, columns=list(range(len(theta_init))))
    df_thetas.loc[0] = theta_init

    for i in range(1, num_optims+1):
        t0 = time.time()
        cell.fit(fobj, f, theta_init, model, bound, options)
        t1 = time.time()

        # train result
        theta, fun_train, cc_train, evar_train = get_results(cell)
        # test result
        fun_test, cc_test, evar_test = compute_func_values(cell,theta,model,fobj,f,options,False)
        theta_init = theta

        if num_optims < 100:
            print("%30.2f %17.2f %17.5f %17.5f %17.5f %17.5f %17.5f" %( (i/num_optims * 100),(t1-t0),fun_train,cc_train,evar_train,cc_test,evar_test))

        output = [(i/num_optims * 100),(t1-t0),fun_train,cc_train,evar_train,cc_test,evar_test]
        df.loc[i] = output
        df_thetas.loc[i] = theta

    print("\n")
    save_results(cell, cell_num, theta, fun_train, cc_train, evar_train,fun_test, cc_test, evar_test)
    df.to_csv(cell_num+'.csv', sep='\t')
    df_thetas.to_csv(cell_num+'_thetas.csv', sep='\t')

    return cell


def save_results(cell, cell_num, theta, fun_train, cc_train, evar_train, fun_test, cc_test, evar_test):
    '''
    Save optimization results
    '''

    cell.result["theta"] = theta
    cell.result["fun"] = fun_train
    cell.result["corrcoef"] = cc_train
    cell.result["evar"] = evar_train

    cell.result["fun_test"] = fun_test
    cell.result["corrcoef_test"] = cc_test
    cell.result["evar_test"] = evar_test

    cell.saveresult(cell_num+'_results.pickle')

    return


def get_results(cell):
    '''
    Get optimization results
        theta
        fun
        corrcoef
        evar
    '''
    result = cell.result

    theta = result["theta"]
    fun = result["fun"]
    corrcoef = result["corrcoef"]
    evar = result["evar"]

    return theta, fun, corrcoef, evar


def compute_func_values(cell, theta, model, fobj, f, options, istrain):
    '''
    Compute the objective function value, correlation, explained variance

    Outputs
    -------
    fun (double):
        objective value

    cc (double):
        correlation coefficient

    ev (double):
        explained variance
    '''
    if istrain:
        data = ccls.get_data(cell, model, options)
    else:
        temp_options = {}
        for key in options.keys():
            if key == 'crossval':
                temp_options[key] = False
            else:
                temp_options[key] = options[key]
        data = ccls.get_data(cell, model, temp_options)

    fun = fobj(theta, data[0], data[1], options)

    if model.lower() == 'lnks_mp':
        y = data[1][1]
        v, y_est = f(theta, data[0], options['pathway'])
    else:
        y = data[1]
        y_est = f(theta, data[0], options['pathway'])

    if istrain:
        cc = stats.corrcoef(y, y_est)
        ev = stats.variance_explained(y, y_est)
    else:
        cc = stats.corrcoef(y[-20000:], y_est[-20000:])
        ev = stats.variance_explained(y[-20000:], y_est[-20000:])

    return fun, cc, ev


def get_env():
    '''
    Assigning input environmental variables to parameters
    '''
    return sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],  sys.argv[5],  sys.argv[6],  sys.argv[7],  sys.argv[8],  sys.argv[9]


def get_initial(model, init_num):
    '''
    Get the initial parameters for model optimization
    '''
    if model.lower() in ["lnks", "lnks_mp"]:
        # LNK initials
        filename = PATH_INITIAL_LNK + init_num[0] + '.initial'
        theta_init_LNK = get_initial_helper(filename)

        # S initials
        filename = PATH_INITIAL_S + init_num[1] + '.initial'
        theta_init_S = get_initial_helper(filename)

        # combine LNK and S initials
        theta_init = np.concatenate((theta_init_LNK,theta_init_S),axis=0)

    elif model.lower() == "lnk":
        # LNK initials
        filename = PATH_INITIAL_LNK + init_num[0] + '.initial'
        theta_init = get_initial_helper(filename)

    elif model.lower() == "spiking":
        # S initials
        filename = PATH_INITIAL_S + init_num[1] + '.initial'
        theta_init = get_initial_helper(filename)

    else:
        raise ValueError('The model name is not appropriate.')

    return theta_init


def get_initial_helper(filename):
    '''
    Load initial parameter
    '''
    fileObject = open(filename, 'rb')
    initials = pickle.load(fileObject)
    fileObject.close()
    theta_init = initials

    return theta_init


def print_results(cell, cell_num, model, objective, pathway, init_num_LNK, init_num_S, num_optims, crossval, is_grad):
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
    corrcoef_test = np.max(cell.result['corrcoef_test'])
    if np.isnan(corrcoef_test) or (corrcoef_test is None):
        corrcoef_test = 0
    evar = np.max(cell.result['evar'])
    if np.isnan(evar) or (evar is None):
        evar = 0
    evar_test = np.max(cell.result['evar_test'])
    if np.isnan(evar_test) or (evar_test is None):
        evar_test = 0
    # fun_init = np.max(cell.result['fun_init'])
    # if np.isnan(fun_init) or (fun_init is None):
    #     fun_init = 0
    # corrcoef_init = np.max(cell.result['corrcoef_init'])
    # if np.isnan(corrcoef_init) or (corrcoef_init is None):
    #     corrcoef_init = 0

    print("\nFit cell %s."  %(cell_num))
    print(" Optimize %s model using %s objective function\n" %(model, objective))
    print(" Optimization options")
    print("\t Cell: %s" %(cell_num))
    print("\t Model: %s" %(model))
    print("\t Objective: %s" %(objective))
    print("\t Initials LNK: %s" %(init_num_LNK))
    print("\t Initials S: %s" %(init_num_S))
    print("\t Pathway: %s" %(pathway))
    print("\t Number of iterations: %s" %(num_optims))
    print("\t Cross-validation: %s" %(crossval))
    print("\t Gradient: %s" %(is_grad))
    print("\n")
    print(" Optimization results")
    print("\t cost function value: %12.4f" % fun)
    print("\t correlation coefficient train: %12.4f" % corrcoef)
    print("\t explained variance train: %12.4f" % evar)
    print("\t correlation coefficient test: %12.4f" % corrcoef_test)
    print("\t explained variance test: %12.4f" % evar_test)
    # print("initial cost function value: %12.4f" % fun_init)
    # print("initial correlation coefficient: %12.4f" % corrcoef_init)


'''
command line function call
'''
if __name__ == '__main__':
    main()

