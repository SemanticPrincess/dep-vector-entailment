% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
TrainModel('splitall', [], 'demo-splitall');

function TrainModel(useAnchor, expName, lambda)

echo "cd home/sick;TrainModel(1, 'anchora', 0.00001);" | /afs/cs/software/bin/matlab_r2013b | tee anchor-00001.txt

echo "cd home/sick;TrainModel(1, 'anchorb', 0.00005);" | /afs/cs/software/bin/matlab_r2013b | tee anchor-00005.txt

echo "cd home/sick;TrainModel(1, 'anchorc', 0.0001);" | /afs/cs/software/bin/matlab_r2013b | tee anchor-00001.txt

echo "cd home/sick;TrainModel(1, 'anchord', 0.000005);" | /afs/cs/software/bin/matlab_r2013b | tee anchor-000005.txt

