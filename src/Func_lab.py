#!/usr/bin/env python3

import os, sys
import subprocess
import json
import time 
pwd = os.path.abspath(os.path.dirname(__file__))

def init(argv):
    package = {'matplotlib': None, 'numpy': None, 'pandas': None, 'scipy': None, 'sympy': None, 'python-ternary': None, 'opencv-python': None}
    out = subprocess.getoutput("pip3 freeze").split('\n')
    tag = False
    for ii in package.keys():
        de = False
        for line in out:
            if line.split('==')[0] == ii:
                package[ii] = line.split('==')[1]
                de = True
        if not de:
            pipcmd = "pip3 install {} --user".format(ii)
            if len(argv)>1:
                if argv[1] == '-s':
                    pipcmd += " -i https://pypi.tuna.tsinghua.edu.cn/simple/"
            so = subprocess.getstatusoutput(pipcmd)
            print("Installing Python Package {}".format(ii))
            if so[0]:
                tag = True
    if not tag:
        with open("{}/config.json".format(os.path.abspath(os.path.dirname(__file__))), 'w') as f1:
            json.dump(package, f1)
        import cv2
        img = cv2.imread('{}/image-src/BPicon.jpg'.format(pwd))
        cv2.imshow("Welcome to BudingPLOT", img)
        cv2.waitKey(1)
        time.sleep(3)
        cv2.destroyAllWindows() 