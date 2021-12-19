import os
import sys
import argparse
import subprocess

old_stdout = sys.stdout
f = open(os.devnull, 'w')

venv_path = '/media/valerio/93f88e81-c907-4ae8-99d0-345b3c02d308/valerio/MBCD/mbcd/venv/bin'
script_path = '/media/valerio/93f88e81-c907-4ae8-99d0-345b3c02d308/valerio/MBCD/mbcd/experiments/mbcd_run.py'
weight_path = '/media/valerio/93f88e81-c907-4ae8-99d0-345b3c02d308/valerio/MBCD/mbcd/weights'
algo_param = '-algo mbcd'
roll_param = ''

parser = argparse.ArgumentParser()
parser.add_argument('-roll', dest='roll', nargs="*", required=True, type=str, help="Experiment scheduler. Pass a sequence of rollout experiments like [mbcd, m2ac], separated by a \n", default=['mbcd'])
parser.add_argument('-iter', dest='iter', required=True, type=int, help="Number of iterations\n")
args = parser.parse_args()

print("This sequence of experiments {} will be executed for {} iterations each".format(args.roll, args.iter))

print("Starting the experiments")

for roll in args.roll:
	roll_param = '-roll ' + roll
	command = ' '.join([venv_path, script_path, algo_param, roll_param])
	for i in range(args.iter):
		print("Started: Experiment {} | Iteration {}".format(roll, i+1))
		print("Command executed: {}".format(command))
		subprocess.run(command, shell=True)
		print("Ended: Experiment {} | Iteration {} \n".format(roll, i))
