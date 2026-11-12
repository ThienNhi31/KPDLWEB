"""Run a project script with blind-file reads blocked before file access.

Usage: python train_only.py stacked_absa.py fit [training arguments...]
Importing this module also enables the guard for a train-only notebook.
"""
import os
import runpy
import sys


def _audit(event,args):
    if event!='open' or not args:return
    name=args[0]
    if isinstance(name,(str,bytes,os.PathLike)):
        basename=os.fsdecode(name).replace('\\','/').rsplit('/',1)[-1]
        if basename.casefold()=='test_blind.csv':
            raise PermissionError('Train-only mode: the user forbids access to test_blind.csv')


sys.addaudithook(_audit)

if __name__=='__main__':
    if len(sys.argv)<2:raise SystemExit('Usage: python train_only.py SCRIPT [arguments...]')
    script=sys.argv[1]
    sys.argv=sys.argv[1:]
    runpy.run_path(script,run_name='__main__')
