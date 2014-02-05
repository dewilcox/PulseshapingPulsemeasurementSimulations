#!/bin/bash

# first part
ipython MasterScript.py 0 > script_output0.txt 2>&1

# second part
ipython MasterScript.py 1 > script_output1.txt 2>&1

# third part
ipython MasterScript.py 2 > script_output2.txt 2>&1

# fourth part
ipython MasterScript.py 3 > script_output3.txt 2>&1

