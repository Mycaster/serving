## test tensorflow model

#### 1. setup tfserving and load tf model
```
. scripts/tf/start-service.sh
```
#### 2. run test script
```
cd scripts/tf/test/ && python transformer_test.py
```


## test torch model
#### 1. export torch model
```
cd scripts/torch/test/ && python torch_model_export.py
```

#### 2. setup tfserving and load torch model
```
. scripts/torch/start-service.sh
```
#### 3. run test script
```
cd scripts/torch/test/ && python resnet.py
```
