# Banana (orange and apple) detection and localization in images using YOLOv8

Install requirements with mentioned command below.
```
pip install -r requirements.txt
```

## Usage

### Training
```
# Training on cpu
python main.py --train --dataset fruits-images/data.yaml --device cpu

# Training on cuda device, i.e. 0 or 0,1,2,3
python main.py --train --dataset fruits-images/data.yaml --device 0
```

You can change number of epochs and batch size by using arguments bellow on training command
```
--epochs 100 --batch-size 16
```

Also, to change the weights used by the model to **train**, use the argument bellow on training command
```
--weights yolov8m.pt
```

### Inference
```
# Test/inference on cpu
python main.py --source fruits-images/test/ --device cpu

# Training on cuda device, i.e. 0 or 0,1,2,3
python main.py --source fruits-images/test/ --device 0
```

You can change the object confidence threshold for detection by using the argument bellow on inference command
```
--conf 0.25
```

Also, to change the weights used by the model to **predict**, use the argument bellow on inference command
```
--weights yolov8m.pt
```

To show and save result image with the predictions, use the argument bellow on inference command
```
--show-results
```
