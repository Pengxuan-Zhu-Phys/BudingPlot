#!/usr/bin/env python3

import os, sys
import subprocess
import json
def init():
    package = {'matplotlib': None, 'numpy': None, 'pandas': None, 'scipy': None, 'sympy': None}
    out = subprocess.getoutput("pip3 freeze").split('\n')
    for ii in package.keys():
        de = False
        for line in out:
            if line.split('==')[0] == ii:
                package[ii] = line.split('==')[1]
                de = True
        if not de:
            os.system("pip3 install {} --user".format(ii))
            print("Installing Python Package {}".format(ii))
    with open("{}/src/config.json".format(os.path.abspath(os.path.dirname(__file__))), 'w') as f1:
        json.dump(package, f1)
pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append("{}/src/".format(pwd))
from buding_plot import Figure


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'init':
            init()
    