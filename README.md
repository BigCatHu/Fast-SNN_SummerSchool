# Fast-SNN

`ZJU Summer School`

参考Fast-SNN的GitHub训练流程 [https://github.com/yangfan-hu/Fast-SNN]

### MNIST/LFW

#### Architectures

For network architectures, we currently support AlexNet, VGG11 we quantize both weights and activations. 

#### Dataset

The dataset MNIST is supposed to be in a 'mnist' folder at the same lavel of 'main.py'
The dataset LFW is constructed after processing by 'lfwdataset.py'  filtering.

#### Train Quantized ANNs

We progressively train full precision, 4, 3, and 2 bit ANN models.

An example to train AlexNet:

```
python MNIST/main.py --arch alex --bit 32 --wd 5e-4
python MNIST/main.py --arch alex --bit 4 --wd 1e-4  --lr 4e-2 --init result/alex_32bit/model_best.pth.tar
python MNIST/main.py --arch alex --bit 3 --wd 1e-4  --lr 4e-2 --init result/alex_4bit/model_best.pth.tar
python MNIST/main.py --arch alex --bit 2 --wd 3e-5  --lr 4e-2 --init result/alex_3bit/model_best.pth.tar
```

#### Evaluate Converted SNNs

The time steps of SNNs are automatically calculated from activation precision, i.e., T = 2^b-1. By default, we use signed IF neuron model.

```
optinal arguments:
    --u                    Use unsigned IF neuron model
```

Example: AlexNet(SNN) performance with traditional unsigned IF neuron model. An 3/2-bit ANN is converted to an SNN with T=3/7.

```
python MNIST/snn.py --arch alex --bit 3 -e -u --init result/alex_3bit/model_best.pth.tar
python MNIST/snn.py --arch alex --bit 2 -e -u --init result/alex_2bit/model_best.pth.tar
```

Example: AlexNet(SNN) performance with signed IF neuron model. An 3/2-bit ANN is converted to an SNN with T=3/7.

```
python MNIST/snn.py --arch alex --bit 3 -e --init result/alex_3bit/model_best.pth.tar
python MNIST/snn.py --arch alex --bit 2 -e --init result/alex_2bit/model_best.pth.tar
```

#### Fine-tune Converted SNNs

By default, we use signed IF neuron model during fine-tuning.

```
optinal arguments:
    --num_epochs / -n               Number of epochs to fine-tune at each layer
                                    default: 1
    --force                         Always update fine-tuned parameters without evaluation on training data
```

Example: finetune converted SNN models.

```
python MNIST/snn_ft.py --arch alex --bit 3 --force --init result/alex_3bit/model_best.pth.tar
python MNIST/snn_ft.py --arch alex --bit 2 --force --init result/alex_2bit/model_best.pth.tar

python LFW/snn_ft.py --arch alex --bit 4 --force --init result_LFW/alex_4bit/model_best.pth.tar
python LFW/snn_ft.py --arch alex --bit 3 --force --init result_LFW/alex_3bit/model_best.pth.tar
```

