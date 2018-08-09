# submit via qsub -t 1-4 -q test_core.q example_4_cluster_submit_me.sh

#$ -cwd
#$ -o $JOB_ID-$TASK_ID.o
#$ -e $JOB_ID-$TASK_ID.e

# enter the virtual environment
source ~sfalkner/virtualenvs/HpBandSter_tests/bin/activate


if [ $SGE_TASK_ID -eq 1]
	then python3 example_4_cluster.py --run_id $JOB_ID --nic_name eth0 --working_dir .
else 
	python3 example_4_cluster.py --run_id $JOB_ID --nic_name eth0  --working_dir . --worker
fi
