echo "========================================"
echo "=======Training res 50 started======"
python train-res50.py
echo "========================================"

echo "=======Training vgg 16 started======"
python train-vgg.py
echo "========================================"

echo "=======Training miniCnn started======"
python train-miniCnn.py
