'''
Fake input output data for a simple temporal problem

The input and output are binary temporal sequences. The output can be predicted using 'n' consecutive input bits. The problem
is that of finding the truth table that relates 'n' input bits to outputs
'''
import numpy as np
import pdb

def fake_data(L, truth_table):
    '''
    inputs:
    ------
    L:  int,
        length of sequence to generate

    n:  int,
        number of binary inputs needed to predict the output

    truth_table:    dictionary
                    map between input sequence and output

    '''
    n = int(np.log2(len(truth_table)))

    input_seq = ''.join(str(np.random.randint(0,2)) for r in range(L+n-1))

    #output_seq = map(lambda tt, seq, s, n: tt[int(seq[s:s+n],2)], [truth_table]*L, input_seq, range(L), [n]*L)
    output_seq = list( map(lambda tt, seq, s, n: tt[int(seq[s:s+n],2)], [truth_table]*L, [input_seq]*L, range(L), [n]*L) )

    return input_seq, output_seq

def truth_table(n):
    '''
    Define a random truth table between n binary inputs and a binary output
    
    inputs:
    ------
    n:  int
        number of input digits into the truth table

    output:
    ------
        dictionary mapping binary inputs to output

    notes:
        with n bits, 2**n-1 is the largest number that can be generated.

    '''

    output = {}
    for i in range(2**n):
        output[i] = np.random.randint(0,2)

    return output
