#!/bin/sh

#PBS -N IBM2
#PBS -l walltime=09:30:00
#PBS -l nodes=1:ppn=1
#PBS -q short
#PBS -e suberror1.file
#PBS -o output.file

cd $PBS_O_WORKDIR

module load scripts
module load Python/3.5.2-intel-2016b
chmod 770 HPCAnalysis.py

python HPCAnalysis.py $cost $x $rep $directed $departure