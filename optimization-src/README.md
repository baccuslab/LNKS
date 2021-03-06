# LNKS model Optimization 
## Setup
Create a folder anywhere you want to keep all optimized results, something like `LNKS-optimized`.
Create initial folders for keeping LNK(`initial_LNK`) and S(`initial_S`) model initial parameters.
The names of these initial folders are fixed in the code, so make sure to keep them same.

    $ dirpath=/path/to/your/LNKS-optimized
    $ mkdir $dirpath
    $ cd $dirpath
    $ mkdir initial_LNK
    $ mkdir initial_S
    
Copy the `fitmodel.py` file to the `$dirname` folder(ex. `LNKS-optimized`).

## Variables
The `fitmodel.py` takes input variables, `cell_num`, `model`, `objective`, `init_num_LNK`, `init_num_S`, `num_optims`, `pathway`, `crossval`, `is_grad`, `bnd_mode`, `gamma`. 
The model and objective function can be selected from `LNK`, `LNKS`, `LNKS_MP`, `Spiking`.
An example of environmental variables setup is shown below:

    $ cell_num=g1
    $ model=LNK
    $ objective=LNK
    $ init_num_LNK=g2
    $ init_num_S=g3
    $ num_optims=10000
    $ pathway=2
    $ crossval=True
    $ is_grad=True
    $ bnd_mode=0
    $ gamma=0.5

## Optimization
Run `fitmodel.py`. Examples are shown below.

### Simple Run
Run `fitmodel.py` in command line as below.

    $ ./fitmodel.py  $cell_num $model $objective $pathway $init_num_LNK $init_num_S $num_optims $crossval $is_grad $bnd_mode $gamma

### Long running jobs
Run long jobs in background and save printed results to a file.

    $ nohup ./fitmodel.py  $cell_num $model $objective $pathway $init_num_LNK $init_num_S $num_optims $crossval $is_grad $bnd_mode $gamma > $cell_num.$init_num_LNK.out &

### Parallel processing
Run multiple optimizations using <a href="http://www.gnu.org/software/parallel" target="_blank">gnu-parallel</a>. The implementation can be found <a href="https://github.com/bongsoos/gnu-parallel" target="_blank">here</a>.
