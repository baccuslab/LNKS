# Linear-Nonlinear-Kinetics-Spiking(LNKS) model
The LNKS model is used in the lab to understand the computations and properties of contrast adaptation in the retina. The model consist of a cascaded connection of four model blocks - linear filter, nonlinearity(threshold, saturation), a first-order dynamic system(markov chain like transition), and a nonlinearity with a feedback. The four blocks have close connection to the biophysical mechanisms in the retina, allowing us to discover where and how the adaptive properties arise in the retinal circuitry.

This repository keeps the tools and examples for using Linear-Nonlinear-Kinetics-Spiking(LNKS) model, also including tools for  Linear-Nonlinear-Kinetics(LNK) model as well.

<!-- Reference: [Ozuysal and Baccus, Neuron, 2012](http://www.sciencedirect.com/science/article/pii/S0896627312000797) -->
Reference of LNK model and contrast adaptation: <a href="http://www.sciencedirect.com/science/article/pii/S0896627312000797" target="_blank">Ozuysal and Baccus, Neuron 2012</a>

## Installation
Clone the repo, and compile the c-extension

    $ git clone https://github.com/baccuslab/LNKS.git
    $ cd LNKS
    $ ./setup
  
Add `LNKS` directory path to the `PYTHONPATH`.

## Tutorial
Please see the examples in the `examples` sub-directory.
