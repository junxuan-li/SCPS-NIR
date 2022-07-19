# downloading precomputed DiLiGenT models
echo "downloading precomputed models"
path="runs/"
mkdir -p $path
cd $path
echo "Start downloading ..."
wget https://www.dropbox.com/s/dws5u3984uw942s/precomputed_models.zip
echo "done!"
cd ..
# The precomputed models is downloaded.
# Run "test_precomputed_results.sh" to get the tested results