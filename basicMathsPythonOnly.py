total_sample_size = 3000000
max_val = 10000

from optparse import Values
import random
from tkinter import Y
from unittest import result

#########################_______ GENERATE DATA FOR ADDITION, SUBTRACTION, MULTIPLICATION AND DIVISION_________########################
# 1. generate two Values
# 2. symbol is +,-,*,/
# 3. str_in = x symb Y
# 4. str out is result
# 5. write to file but not in method

def generate_math_pairs(operation):
    x = random.randrange(1,max_val)
    y = random.randrange(1,max_val)

    if operation == 'add':
