# apg-em
Expectation Maximization EM model for APG subject IARFID UPV 

# Usage
usage: test.py [-h] [-g1 G1 [G1 ...]] [-g2 G2 [G2 ...]] I N {task1,task2,base}

    Reproduce an example similar to the previous example with two
    unidimensional distributions.
    -TASK1 - with equal and known VARIANCE where the MEANS and π1, π2 are unknown.
    -TASK2 - with equal and known MEAN where the VARIANCES and π1, π2 are unknown.
    -BASE - with equal VARIANCES and and known MEAN where π1, π2 are unknown.
    

positional arguments:

    I                   number of iterations of EM algorithm
    N                   number of data to be generated
    {task1,task2,base}  select task to be reproduced

optional arguments:

    -h, --help          show this help message and exit
    -g1 G1 [G1 ...]     Specify base gaussian 1 and its π. Format: π1 mean1 var1
    -g2 G2 [G2 ...]     Specify base gaussian 2 and its π. Format: π2 mean2 var2

