# yolov5及其改进

# **工业缺陷检测顶会论文常用数据集**

## 参考：

- https://github.com/Charmve/Surface-Defect-Detection/blob/master/ReadmeChinese.md

- https://www.cnblogs.com/lky-learning/p/13622271.html
- `deepseek`，`chatgpt`

|                 数据集                  |                             特点                             |                             链接                             |
| :-------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|         钢材表面数据集：NEU-CLS         | 数据集收集了夹杂、划痕、压入氧化皮、裂纹、麻点和斑块6种缺陷，每种缺陷300张，图像尺寸为200×200。 | http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/ |
| 弱监督学习下的工业光学检测（DAGM 2007） | 主要针对纹理背景上的杂项缺陷。 较弱监督的训练数据。 包含十个数据集，前六个为训练数据集，后四个为测试数据集。 |         https://hci.iwr.uni-heidelberg.de/node/3616          |
|   Kaggle - 钢材表面数据集：Severstal    |                              \                               |  https://www.kaggle.com/c/severstal-steel-defect-detection   |
|       金属表面数据集：KolektorSDD       |                              \                               | https://pan.baidu.com/share/init?surl=HSzHC1ltHvt1hSJh_IY4Jg |
|          MVTec 异常检测数据集           |                              \                               |        http://www.mvtec.com/company/research/datasets        |

# yolo训练

## 模型检测

**相关参数**

- `weights`：训练好的模型文件

  ```
  python detect.py --weights yolov5s.pt
  ```

- `source`：检测的目标

  ```
  python detect.py --weights yolov5s.pt --source (img.jpg screen)
  ```

- `conf-thres` 置信度阈值，越低框越多

  ```
  python detect.py --weights yolov5s.pt --conf-thres 0.8
  ```

- `iou-thres` 置信度阈值，越高框越多

## 数据集构建

```
NEU-DET
  ├── images
  │   ├── train     # 训练集图像
  │   └── val       # 验证集图像
  └── labels
      ├── train     # 训练集标签（TXT文件）
      └── val       # 验证集标签（TXT文件）
```

- **关键点**：确保`images`和`labels`目录下对应文件名一致（如`001.jpg`与`001.txt`）。

```
labelimg
```

## 模型训练

![1](./picture/cuowu.png)

- 采用数据集：`NEU surface defect database`

```
path: ./NEU-DET
train: images/train
val: images/val

nc: 6
names: ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
```

```
names:
  0: Crazing    # 裂纹
  1: Inclusion  # 夹杂物
  2: Patches    # 斑块
  3: Pitted     # 点蚀
  4: Scratches  # 划痕
  5: Rolled     # 轧制缺陷
```

- 修改`train.py`

```
    parser.add_argument('--data', type=str, default=ROOT / 'NEU-DET/data.yaml', help='dataset.yaml path')

	parser.add_argument('--project', default=ROOT / 'NEU-DET/output', help='save to project/name')
	
	parser.add_argument('--epochs', type=int, default=1, help='total training epochs')

```

- 结果

```
输出在output;
查看：
tensorboard --logdir NEU-DET\output
```

# 评估

- `epoch`
  - 轮次：一次完整的遍历和训练 

- `iteration`
  - 一次迭代（iteration）指模型处理数量为 --batch-size  的数据并完成一次参数更新 

## baseline

`NEU-DET\output\exp3`

`fps=62`

| epoch | batch-size |  iteration  | train | val  |
| :---: | :--------: | :---------: | :---: | :--: |
|  50   |     16     | 107*50=5350 | 1711  |  30  |

```
Model summary: 157 layers, 7026307 parameters, 0 gradients, 15.8 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 
                   all         30         64      0.828      0.723       0.82      0.457
               crazing         30          8      0.793      0.375      0.668      0.255
             inclusion         30         15      0.659      0.733       0.77      0.286
               patches         30         17      0.842      0.942      0.948      0.628
        pitted_surface         30          8       0.96      0.875      0.982      0.739
       rolled-in_scale         30          9      0.829      0.556      0.681      0.337
             scratches         30          7      0.884      0.857      0.869      0.494
```

- **精确率（`Precision`）**：预测为正样本中实际为正的比例（`TP/(TP+FP)`），关注误检率；
- **召回率**（`Recall`）：实际正样本中被正确预测的比例（`TP/(TP+FN)`），关注漏检率
- **`mAP@0.5`**：以交并比（`IoU`）阈值为0.5计算的精度，反映粗粒度检测能力；
- **`mAP@0.5:0.95`**：`IoU`阈值从0.5到0.95的平均精度，衡量精细定位能力
- `FPS`：https://blog.csdn.net/m0_56247038/article/details/126673489

## 改进方法

- NEU-DET 数据集特点：工业表面缺陷检测场景，可能存在**小目标、复杂背景**等问题

### 1. **替换特征金字塔网络**(未成功实现)

> - **方法**：将默认的 **FPN+PAN 结构** 替换为 **AF-FPN** 或 **Bi-FPN**。
> - **原理**：
>   - **AF-FPN**：通过自适应注意力模块（AAM）和特征增强模块（FEM）减少特征图信息丢失，提升小目标检测能力。
>   - **Bi-FPN**：通过双向跨尺度连接和加权特征融合，强化不同层次特征的语义一致性。

- 替换为AF-FPN

### 2. **CBAM**

> - **原理**：引入空间注意力机制,通过通道注意力（关注重要特征通道）和空间注意力（聚焦目标区域），提升对缺陷区域的敏感度，抑制复杂背景干扰。

1. `models/common.py`中加入以下模块

   ```
   class ChannelAttention(nn.Module):
       def __init__(self, in_planes, ratio=16):
           super(ChannelAttention, self).__init__()
           self.avg_pool = nn.AdaptiveAvgPool2d(1)
           self.max_pool = nn.AdaptiveMaxPool2d(1)
    
           self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
           self.relu = nn.ReLU()
           self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
    
           self.sigmoid = nn.Sigmoid()
    
       def forward(self, x):
           avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
           max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
           out = self.sigmoid(avg_out + max_out)
           return out
    
    
   class SpatialAttention(nn.Module):
       def __init__(self, kernel_size=7):
           super(SpatialAttention, self).__init__()
    
           assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
           padding = 3 if kernel_size == 7 else 1
    
           self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
           self.sigmoid = nn.Sigmoid()
    
       def forward(self, x):
           avg_out = torch.mean(x, dim=1, keepdim=True)
           max_out, _ = torch.max(x, dim=1, keepdim=True)
           x = torch.cat([avg_out, max_out], dim=1)
           x = self.conv(x)
           return self.sigmoid(x)
    
    
    
   class CBAM(nn.Module):
       def __init__(self, c1, c2):
           super(CBAM, self).__init__()
           self.channel_attention = ChannelAttention(c1)
           self.spatial_attention = SpatialAttention()
    
       def forward(self, x):
           out = self.channel_attention(x) * x
           out = self.spatial_attention(out) * out
           return out
   ```

2. `models/yolo.py`中的`parse_model`加入末尾的CBAM

   <img src="./picture/shili.png" alt="1742979316736" style="zoom:67%;" />

3. `models`下新建`yolov5s_CBAM.yaml`

   ```
   # YOLOv5 🚀 by YOLOAir, GPL-3.0 license
    
   # Parameters
   nc: 6  # number of classes
   depth_multiple: 0.33  # model depth multiple
   width_multiple: 0.50  # layer channel multiple
   anchors:
     - [10,13, 16,30, 33,23]  # P3/8
     - [30,61, 62,45, 59,119]  # P4/16
     - [116,90, 156,198, 373,326]  # P5/32
    
   # YOLOv5 v6.0 backbone
   backbone:
     # [from, number, module, args]
     [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
      [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
      [-1, 3, C3, [128]],
      [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
      [-1, 6, C3, [256]],
      [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
      [-1, 9, C3, [512]],
      [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
      [-1, 3, C3, [1024]],
      [-1, 1, SPPF, [1024, 5]],  # 9
     ]
    
   # YOLOv5 v6.0 head
   head:
     [[-1, 1, Conv, [512, 1, 1]],
      [-1, 1, nn.Upsample, [None, 2, 'nearest']],
      [[-1, 6], 1, Concat, [1]],  # cat backbone P4
      [-1, 3, C3, [512, False]],  # 13
    
      [-1, 1, Conv, [256, 1, 1]],
      [-1, 1, nn.Upsample, [None, 2, 'nearest']],
      [[-1, 4], 1, Concat, [1]],  # cat backbone P3
      [-1, 3, C3, [256, False]],  # 17 (P3/8-small)
      [-1, 1, CBAM, [256]],
    
      [-1, 1, Conv, [256, 3, 2]],
      [[-1, 14], 1, Concat, [1]],  # cat head P4
      [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)
      [-1, 1, CBAM, [512]],
    
      [-1, 1, Conv, [512, 3, 2]],
      [[-1, 10], 1, Concat, [1]],  # cat head P5
      [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)
      [-1, 1, CBAM, [1024]],
    
      [[18, 21, 24], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
     ]
   ```

4. 调整参数`train.py`

   ```
   parser.add_argument('--cfg', type=str, default=ROOT / 'models/yolov5s_CBAM.yaml', help='model.yaml path')
   ```

- 运行效果：``NEU-DET\output\exp16`

  ```
  YOLOv5s_CBAM summary: 190 layers, 7069609 parameters, 0 gradients, 15.9 GFLOPs
  Class            Images  Instances     P          R      mAP50   mAP50-95
   all              30         64      0.701      0.789      0.804       0.42
   crazing          30          8      0.605       0.25      0.405      0.138
  inclusion         30         15      0.596      0.867      0.826       0.44
    patches         30         17      0.807      0.984      0.928      0.552
  pitted_surface    30          8      0.923          1      0.995       0.67
  rolled-in_scale   30          9      0.626      0.778      0.812      0.286
   scratches        30          7      0.647      0.857      0.858      0.436
  ```

`fps=42.55`

### 3. SENet

> - **原理**： 通过**通道注意力机制**，采用**Squeeze-Excitation**结构，动态学习通道权重。 

1. `models/common.py`中加入以下模块

   ```
   class SEAttention(nn.Module):
    
       def __init__(self, channel=512, reduction=16):
           super().__init__()
           self.avg_pool = nn.AdaptiveAvgPool2d(1)
           self.fc = nn.Sequential(
               nn.Linear(channel, channel // reduction, bias=False),
               nn.ReLU(inplace=True),
               nn.Linear(channel // reduction, channel, bias=False),
               nn.Sigmoid()
           )
    
       def forward(self, x):
           b, c, _, _ = x.size()
           y = self.avg_pool(x).view(b, c)
           y = self.fc(y).view(b, c, 1, 1)
           return x * y.expand_as(x)
   ```

2. `models/yolo.py`中的`parse_model`加入

   ```
   elif m in {SEAttention}:
               c1,c2= ch[f],args[0]
               if c2 != no:
                   c2 = make_divisible(c2 * gw, 8)
               args = [c1, *args[1:]]
   ```

   

3. `models`下新建`yolov5s_SEnet.yaml`

   ```
   # YOLOv5 🚀 by YOLOAir, GPL-3.0 license
    
   # Parameters
   nc: 6  # number of classes
   depth_multiple: 0.33  # model depth multiple
   width_multiple: 0.50  # layer channel multiple
   anchors:
     - [10,13, 16,30, 33,23]  # P3/8
     - [30,61, 62,45, 59,119]  # P4/16
     - [116,90, 156,198, 373,326]  # P5/32
    
   # YOLOv5 v6.0 backbone
   backbone:
     # [from, number, module, args]
     [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
      [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
      [-1, 3, C3, [128]],
      [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
      [-1, 6, C3, [256]],
      [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
      [-1, 9, C3, [512]],
      [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
      [-1, 3, C3, [1024]],
      [-1, 1, SPPF, [1024, 5]],  # 9
     ]
    
   # YOLOv5 v6.0 head
   head:
     [[-1, 1, Conv, [512, 1, 1]],
      [-1, 1, nn.Upsample, [None, 2, 'nearest']],
      [[-1, 6], 1, Concat, [1]],  # cat backbone P4
      [-1, 3, C3, [512, False]],  # 13
    
      [-1, 1, Conv, [256, 1, 1]],
      [-1, 1, nn.Upsample, [None, 2, 'nearest']],
      [[-1, 4], 1, Concat, [1]],  # cat backbone P3
      [-1, 3, C3, [256, False]],  # 17 (P3/8-small)
      [-1, 1, SEAttention, [256]],
    
      [-1, 1, Conv, [256, 3, 2]],
      [[-1, 14], 1, Concat, [1]],  # cat head P4
      [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)
      [-1, 1, SEAttention, [512]],
    
      [-1, 1, Conv, [512, 3, 2]],
      [[-1, 10], 1, Concat, [1]],  # cat head P5
      [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)
      [-1, 1, SEAttention, [1024]],
    
      [[18, 21, 24], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
     ]
   ```

4. 调整参数`train.py`

   ```
       parser.add_argument('--cfg', type=str, default=ROOT / 'models/yolov5s_SENet.yaml', help='model.yaml path')
   
   ```

- 运行效果：`NEU-DET\output\exp17`

  ```
  YOLOv5s_SENet summary: 178 layers, 7069315 parameters, 0 gradients, 15.8 GFLOPs
        Class    Images  Instances       P          R      mAP50   mAP50-95
       all         30         64      0.704      0.772      0.768      0.405
     crazing       30          8      0.451       0.25        0.4      0.096
    inclusion      30         15       0.55      0.815      0.735      0.349
     patches       30         17      0.787      0.941      0.899      0.564
  pitted_surface   30          8      0.941          1      0.995      0.708
  rolled-in_scale  30          9       0.75      0.778      0.794      0.311
      scratches    30          7      0.747      0.846      0.783      0.401
  ```

  `fps=57.8`

### 4. CA

> - **原理**： 结合**通道注意力**与**坐标信息编码**，通过分解全局池化捕获空间坐标特征。 

1. `models/common.py`中加入以下模块

   ```
   #-----------------CA----------------
   
   import torch
   import torch.nn as nn
   import math
   import torch.nn.functional as F
    
   class h_sigmoid(nn.Module):
       def __init__(self, inplace=True):
           super(h_sigmoid, self).__init__()
           self.relu = nn.ReLU6(inplace=inplace)
    
       def forward(self, x):
           return self.relu(x + 3) / 6
    
   class h_swish(nn.Module):
       def __init__(self, inplace=True):
           super(h_swish, self).__init__()
           self.sigmoid = h_sigmoid(inplace=inplace)
    
       def forward(self, x):
           return x * self.sigmoid(x)
    
   class CoordAtt(nn.Module):
       def __init__(self, inp, oup, reduction=32):
           super(CoordAtt, self).__init__()
           self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
           self.pool_w = nn.AdaptiveAvgPool2d((1, None))
    
           mip = max(8, inp // reduction)
    
           self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
           self.bn1 = nn.BatchNorm2d(mip)
           self.act = h_swish()
           
           self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
           self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
           
    
       def forward(self, x):
           identity = x
           
           n,c,h,w = x.size()
           x_h = self.pool_h(x)
           x_w = self.pool_w(x).permute(0, 1, 3, 2)
    
           y = torch.cat([x_h, x_w], dim=2)
           y = self.conv1(y)
           y = self.bn1(y)
           y = self.act(y) 
           
           x_h, x_w = torch.split(y, [h, w], dim=2)
           x_w = x_w.permute(0, 1, 3, 2)
    
           a_h = self.conv_h(x_h).sigmoid()
           a_w = self.conv_w(x_w).sigmoid()
    
           out = identity * a_w * a_h
    
           return out 
   ```

2. `models/yolo.py`中的`parse_model`加入末尾的

   ![1742991213601](./picture/shilicoordatt.png)

3. `models`下新建`yolov5s_CA.yaml`

   ```
   # YOLOv5 🚀 by YOLOAir, GPL-3.0 license
   
   # Parameters
   nc: 6  # number of classes
   depth_multiple: 0.33  # model depth multiple
   width_multiple: 0.50  # layer channel multiple
   anchors:
     - [10,13, 16,30, 33,23]  # P3/8
     - [30,61, 62,45, 59,119]  # P4/16
     - [116,90, 156,198, 373,326]  # P5/32
    
   # YOLOv5 v6.0 backbone
   backbone:
     # [from, number, module, args]
     [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
      [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
      [-1, 3, C3, [128]],
      [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
      [-1, 6, C3, [256]],
      [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
      [-1, 9, C3, [512]],
      [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
      [-1, 3, C3, [1024]],
      [-1,1,CoordAtt,[1024]], # CA
      [-1, 1, SPPF, [1024, 5]],  # 9
     ]
    
   # YOLOv5 v6.0 head
   head:
     [[-1, 1, Conv, [512, 1, 1]],
      [-1, 1, nn.Upsample, [None, 2, 'nearest']],
      [[-1, 6], 1, Concat, [1]],  # cat backbone P4
      [-1, 3, C3, [512, False]],  # 13
    
      [-1, 1, Conv, [256, 1, 1]],
      [-1, 1, nn.Upsample, [None, 2, 'nearest']],
      [[-1, 4], 1, Concat, [1]],  # cat backbone P3
      [-1, 3, C3, [256, False]],  # 17 (P3/8-small)
      #[-1, 1, CoordAtt, [256]],
    
      [-1, 1, Conv, [256, 3, 2]],
      [[-1, 15], 1, Concat, [1]],  # cat head P4
      [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)
      #[-1, 1, CoordAtt, [512]],
    
      [-1, 1, Conv, [512, 3, 2]],
      [[-1, 11], 1, Concat, [1]],  # cat head P5
      [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)
      #[-1, 1, CoordAtt, [1024]],
    
      [[18, 22, 24], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
     ]
   ```

4. 调整参数`train.py`

   ```
       parser.add_argument('--cfg', type=str, default=ROOT / 'models/yolov5s_CA.yaml', help='model.yaml path')
   ```

- 运行效果：`NEU-DET\output\exp20`

  ```
  YOLOv5s_CA summary: 167 layers, 7051955 parameters, 0 gradients, 15.8 GFLOPs
        Class       Images  Instances       P          R      mAP50   mAP50-95:
        all           30         64      0.677      0.757      0.818      0.442
        crazing       30          8      0.716      0.324      0.676      0.248
        inclusion     30         15      0.576      0.867      0.781       0.37
        patches       30         17      0.827      0.941      0.975      0.603
     pitted_surface   30          8      0.819          1      0.995      0.715
  rolled-in_scale     30          9      0.528      0.556      0.696      0.312
    scratches         30          7      0.595      0.857      0.782      0.405
               
  ```

  `fps=61.3`

## 效果对比

- 这几个模块的添加为什么在有些类上能够带来指标的提升，但是在有些类上会使指标下降，这个可以结合对结果的可视化进行观察，重要的是有理有据；
- 那些折线图表达的是什么含义，与baseline的折线图有什么区别，表示什么，主要看差异比较大的就行了。

`tensorboard --logdir NEU-DET\output`

|          | 位置NEU-DET/output/ |
| -------- | ------------------- |
| baseline | exp3                |
| `CBAM`   | exp16               |
| `SENet`  | exp17               |
| `CA`     | exp20               |

|             缺陷类别              |       频率分类       |                      特点                       |
| :-------------------------------: | :------------------: | :---------------------------------------------: |
|        **斑块（patches）**        |       **高频**       | 局部集中分布、边界清晰，易被注意力机制捕捉**1** |
|       **夹杂（inclusion）**       |       **高频**       |  与背景对比度强，形状不规则但密度差异显著**1**  |
|  **点蚀表面（pitted_surface）**   | 低频（因检测性能高） |           凹坑特征明显，空间分布孤立            |
|        **龟裂（crazing）**        |       **低频**       |           细密裂纹，尺寸小且分布散乱            |
| **轧入氧化皮（rolled-in_scale）** |       **低频**       |       表面氧化层压入，纹理渐变、对比度低        |
|       **划痕（scratches）**       |       **低频**       |           线性表面损伤，易与背景混淆            |

- CBAM
  - 精度变化：斑块（patches）和夹杂（inclusion）精度上升，其他精度有所下降
  - 通道和空间注意力会为高频部分分配更多的权重，抑制低频的背景部分，所以会对`patches`和`inclusion`会比较敏感

- SENet
  - 添加这个模块后所有类别精度均下降，其中**龟裂（crazing）降幅最大（-43%）**，斑块（patches）降幅最小（-6.5%）。
  - 

| 加入模块 | 精度变化                                                     | 原理                                                         | 原因分析                                                     |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| CBAM     | 斑块（patches）和夹杂（inclusion）精度上升，其他精度有所下降 | 引入空间注意力机制,通过通道注意力（关注重要特征通道）和空间注意力（聚焦目标区域），提升对**缺陷区域的敏感度，抑制复杂背景干扰。** | - 通道和空间注意力会为高频部分分配更多的权重，抑制低频的背景部分，所以会对`patches`和`inclusion`会比较敏感 |
| SENet    | 添加这个模块后所有类别精度均下降，其中**龟裂（crazing）降幅最大（-43%）**，斑块（patches）降幅最小（-6.5%）。 |                                                              |                                                              |
| CA       | **整体精度（P）显著下降**：所有类别精度均下降，其中**轧入氧化皮（-36.3%）**和**划痕（-32.7%）**降幅最大。 |                                                              |                                                              |

- baseline   

<img src="./NEU-DET/output/exp3/P_curve.png" alt="1742979316736" style="zoom:67%;" />

- CBAM

<img src="./NEU-DET/output/exp16/P_curve.png" alt="1742979316736" style="zoom:67%;" />

- SENet

<img src="./NEU-DET/output/exp17/P_curve.png" alt="1742979316736" style="zoom:67%;" />

- CA

<img src="./NEU-DET/output/exp20/P_curve.png" alt="1742979316736" style="zoom:67%;" />