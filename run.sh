echo "========================================"
echo "=======Training res 50 started======"
python train-res50.py
echo "========================================"

echo "=======Training vgg 16 started======"
python train-xception.py
echo "========================================"

echo "=======Training miniCnn started======"
python train-miniCnn.py

echo "=======Pre-Training res 50 started======"
python train-res50.py --pretrain yes --name res50pretrain
echo "========================================"

echo "=======Training vgg 16 started======"
python train-xception.py --pretrain yes --name xceptionpretrain
echo "========================================"
