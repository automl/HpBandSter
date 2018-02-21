# submit via qsub -t 1-4 submit_me.sh

#$ -q test_core.q
#$ -cwd
#$ -o logs/$JOB_ID-$TASK_ID.o
#$ -e logs/$JOB_ID-$TASK_ID.e

# enter the virtual environment
source ~sfalkner/virtual_envs/HpBandSter_tests/bin/activate

python3 run_me.py --run_id $JOB_ID --array_id $SGE_TASK_ID  --working_dir .
