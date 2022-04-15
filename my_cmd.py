import os
from subprocess import call
import sys

def run(command):
    try:
        code = call(command, shell=True)
        if code == 0:
            print(f'Command finished with return code {code}: {command}')
        else:
            print(f'Command failed with return code {code}: {command}')
            sys.exit(1)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)
