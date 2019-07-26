ARRAY=(1 2 3 4)

for num in ${ARRAY[@]}; do
    /usr/bin/python3 /home/takumi/research/py-MDNet/tracking/run_tracker.py -v True -f --cuda_num 2

done