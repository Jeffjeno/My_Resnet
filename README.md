# Goal
Replicate  the resnet-50 network from scratch

# the network architecture
I'll use the 3 bottleneck layers in the 34-layer net.
If the dimensions increase,I considered to use the option B (**using the projection shortcut used to match dimensions(done by 1*1 convolutions)**).And the shortcuts are performed with the stride of 2,when going across the feature maps of  the two sizes.

# Structure
![alt text](image.png)

# the structure picture in details 
```plaintext
Input (3 channels, 224x224)
        |
        v
Conv1 (64 channels, 7x7 kernel, stride 2, padding 1)
        |
        v
BatchNorm + ReLU
        |
        v
MaxPooling (3x3 kernel, stride 2, padding 1)
        |
        v
Block1 (3 Bottleneck Layers)
        |
        v
Block2 (4 Bottleneck Layers, first layer with downsampling)
        |
        v
Block3 (6 Bottleneck Layers, first layer with downsampling)
        |
        v
Block4 (3 Bottleneck Layers, first layer with downsampling)
        |
        v
AdaptiveAvgPool (1x1)
        |
        v
Flatten
        |
        v
Fully Connected (2048 -> 1000)
        |
        v
Softmax
```

Here is my resposity:
https://github.com/Jeffjeno/My_Resnet