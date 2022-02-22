import os
import sys
import argparse
import subprocess

old_stdout = sys.stdout
f = open(os.devnull, 'w')

venv_path = '/media/valerio/93f88e81-c907-4ae8-99d0-345b3c02d308/valerio/MBCD/mbcd/venv/bin/python'
script_path = '/media/valerio/93f88e81-c907-4ae8-99d0-345b3c02d308/valerio/MBCD/mbcd/experiments/mbcd_run.py'
weight_path = '/media/valerio/93f88e81-c907-4ae8-99d0-345b3c02d308/valerio/MBCD/mbcd/weights'
algo_param = '-algo mbcd'

parser = argparse.ArgumentParser()
parser.add_argument('-iter', dest='iter', required=True, type=int, help="Number of iterations\n")
args = parser.parse_args()

print("Base MBCD will be executed for {} iterations".format(args.iter))

print("Starting the experiments")

command = ' '.join([venv_path, script_path, algo_param])
for i in range(args.iter):
    print("Started: Iteration {}".format(i + 1))
    print("Command executed: {}".format(command))
    subprocess.run(command, shell=True)
    print("Ended: Iteration {} \n".format(i))
