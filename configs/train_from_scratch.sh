EXP_NUM="diligent"
CUDA_NUM="0"
TESTING="False"
echo $EXP_NUM/cuda:$CUDA_NUM/Testing:$TESTING
python train.py --config configs/$EXP_NUM/harvest.yml --cuda $CUDA_NUM   --testing $TESTING
python train.py --config configs/$EXP_NUM/buddha.yml --cuda $CUDA_NUM   --testing $TESTING
python train.py --config configs/$EXP_NUM/pot1.yml --cuda $CUDA_NUM   --testing $TESTING
python train.py --config configs/$EXP_NUM/pot2.yml --cuda $CUDA_NUM   --testing $TESTING
python train.py --config configs/$EXP_NUM/ball.yml --cuda $CUDA_NUM   --testing $TESTING
python train.py --config configs/$EXP_NUM/reading.yml --cuda $CUDA_NUM   --testing $TESTING
python train.py --config configs/$EXP_NUM/cow.yml --cuda $CUDA_NUM   --testing $TESTING
python train.py --config configs/$EXP_NUM/cat.yml --cuda $CUDA_NUM   --testing $TESTING
python train.py --config configs/$EXP_NUM/bear.yml --cuda $CUDA_NUM   --testing $TESTING
python train.py --config configs/$EXP_NUM/goblet.yml --cuda $CUDA_NUM   --testing $TESTING
