# apg-em
Expectation Maximization EM model for APG subject IARFID UPV 

# Usage
usage: test.py [-h] I N {task1,task2}

    Reproduce an example similar to the previous example with two
    unidimensional distributions.
    -TASK1 - with equal and known VARIANCE where the MEANS and π1, π2 are unknown.
    -TASK2 - with equal and known MEAN where the VARIANCES and π1, π2 are unknown.
    

positional arguments:
  I              number of iterations of EM algorithm
  N              number of data to be generated
  {task1,task2}  select task to be reproduced

optional arguments:
  -h, --help     show this help message and exit

