"""Reproduce Figures 1--6 and 10; Figures 7--9 use the local viewer."""
from pathlib import Path
import os,sys,subprocess
folder=Path(__file__).resolve().parent
for number in [1,2,3,4,5,6,10]:
    print(f'Figure {number}',flush=True)
    subprocess.run([sys.executable,str(folder/f'plot_fig{number}.py')],check=True,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
