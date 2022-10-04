# total_sample_size = 3000000
# max_val = 10000

total_sample_size = 30
max_val = 30

from optparse import Values
import random
from tkinter import Y
from unittest import result
import pandas as pd
import numpy as np


#########################_______ GENERATE DATA FOR ADDITION, SUBTRACTION, MULTIPLICATION AND DIVISION_________########################
# 1. generate two Values
# 2. symbol is +,-,*,/
# 3. str_in = x symb Y
# 4. str out is result
# 5. write to file but not in method

def gen_math_pairs(operation):
    x = random.randrange(1,max_val)
    y = random.randrange(1,max_val)

    if operation == 'add':
        result = x+y
        symbol = '+'
    elif operation == 'sub':
        result = x-y
        symbol ='-'
    elif operation == 'mul':
        result = x*y
        symbol ='*'
    elif operation == 'div':
        result = x/y
        symbol ='/'

    str_in = "{}{}{}\n".format(x,symbol,y)
    str_out = "{}\n".format(result)

    return x,y,result, str_in, str_out


############# generate training data file ###########################

def gen_train_data_n_labels(method = 'add'):
    training_in = np.empty(shape=(total_sample_size, 2), dtype=int)
    labels = np.empty(shape=(total_sample_size, 1), dtype=int)

    for i in range(0,total_sample_size):
        # Creating the first Dataframe using dictionary
        x,y,result,str_in,str_out = gen_math_pairs('add')
        print(str_in, str_out)
        # training_in = pd.DataFrame({"x":[x],
        #                  "y":[y]})

        # for x in range(1, 6):
        # for y in range(1, 6):
        training_in[i] = x, y
        labels[i] = result 

    return training_in, labels


if __name__ == "__main__":
        training_in,labels = gen_train_data_n_labels(method = 'add')
        # print(training_in, labels, type(training_in))
