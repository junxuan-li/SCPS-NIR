# Before runing this file, you need to first run "download_precomputed_results.sh" to download the precomputed models
echo "Start unzipping ..."
path="runs/"
cd $path
unzip -o precomputed_models.zip
cd ..

# run quick testing on precomputed models
CUDA_NUM="0"
TESTING="True"
QUICKTESTING="True"   # Change QUICKTESTING to "False" for more visualization results
echo cuda:$CUDA_NUM/Testing:$TESTING/quick_testing:$QUICKTESTING

echo "Start testing diligent"
FILES="configs/diligent/*.yml"
for f in $FILES
do
  python train.py --config $f --cuda $CUDA_NUM   --testing $TESTING --quick_testing $QUICKTESTING
done

echo "Start testing apple"
FILES="configs/apple/*.yml"
for f in $FILES
do
  python train.py --config $f --cuda $CUDA_NUM   --testing $TESTING --quick_testing $QUICKTESTING
done