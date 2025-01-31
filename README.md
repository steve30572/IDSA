This is the implementation of the paper "Exploiting Inter-Domain Sensor Alignment in Multivariate Time-Series
Unsupervised Domain Adaptation"

Please refer to the requirements.txt for the used library.

You can simply run the code with python script
```
python train.py
```

the arguments are as follow:

- source: place where the source dataset is placed
- target: place where the target dataset is placed
- batch_size
- hidden_size
- epoch
- num_layers: number of layers for temporal layer
- lr: learning rate
- weight_decay
- temp_mode: setting temporal layer to lstm ('lstm') or tcn ('tcn')
- dilation_factor: dilation factor for tcn layer
- coral: $\lambda_2$ in our paper
- inter: $\lambda_1$ in our paper
