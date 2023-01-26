# Zero-Shot Cross-Lingual Intent Prediction and Slot Filling

**Goal**: Train a joint intent prediction and slot filling model using English and generalize to other languages.

### Datasets
MultiAtis++: Please visit https://github.com/amazon-research/multiatis, download the dataset and put it under `./data` directory.

### How to Run: Joint Training

#### joint training (English Only)
Firstly, change the `model_type` variable in `joint_en.py` to `mbert|bert|roberta|xlm-roberta` in order to use different transformer models.
Then run the following command:

```
python3 joint_en.py
```

#### joint training (English Only with Code Switching)

##### Generate Code-Switching training data

Firstly, change the `inFile` variable in `code_switch.py` to the input filename; change the `outFile` variable to the pickle output filename. Then run the following command:

```
python3 code_switch.py
```

##### Run Code-Switching Method
Firstly, change the `code_switch` variable to `1`, then run the following command:
```
python3 joint_en.py
```

### Result

#### MBERT (bert-base-multilingual-uncased)

|languange| Intent Acc.  | Slot F1 |
| ------- | ----------   | ------- |
| ES      |    0.9440    |  0.9595 |
| DE      |    0.9440    |  0.9648 |
| ZH      |    0.8096    |  0.8511 |
| JA      |    0.7290    |  0.7858 |
| PT      |    0.9328    |  0.9561 |
| FR      |    0.9227    |  0.9546 |
| HI      |    0.6909    |  0.7814 |
| TR      |    0.7202    |  0.8761 |
| RW      |    0.2004    |  0.7997 |
| SW      |    0.5408    |  0.8844 |

#### BERT (bert-base-uncased)


|       | Intent Acc.  | Slot F1 |
| ----- | ----------   | ------- |
| ES    |    0.6730    |  0.8351 |
| DE    |    0.0638    |  0.7884 |
| ZH    |    0.1187    |  0.5770 |
| JA    |    0.0660    |  0.4297 |
| PT    |    0.3314    |  0.8217 |
| FR    |    0.4154    |  0.8309 |
| HI    |    0.0324    |  0.3504 |
| TR    |    0.1580    |  0.7090 |
| RW    |    0.0470    |  0.5281 |
| SW    |    0.1780    |  0.5757 |


### Roberta (roberta-base)

|languange| Intent Acc.  | Slot F1 |
| ------- | ----------   | ------- |
| ES      |    0.1019    |  0.7914 |
| DE      |    0.5106    |  0.8810 |
| ZH      |    0.5901    |  0.4906 |
| JA      |    0.3516    |  0.6060 |
| PT      |    0.3292    |  0.8436 |
| FR      |    0.2508    |  0.8455 |
| HI      |    0.0666    |  0.3580 |
| TR      |    0.2797    |  0.7418 |
| RW      |    0.1478    |  0.7384 |
| SW      |    0.1578    |  0.7117 |

### XLM-Roberta (xlm-roberta-base)

|languange| Intent Acc.  | Slot F1 |
| ------- | ----------   | ------- |
| ES      |    0.9216    |  0.9607 |
| DE      |    0.9440    |  0.9711 |
| ZH      |    0.8835    |  0.9219 |
| JA      |    0.8667    |  0.8746 |
| PT      |    0.9182    |  0.9660 |
| FR      |    0.9305    |  0.9487 |
| HI      |    0.8499    |  0.8890 |
| TR      |    0.7664    |  0.8745 |
| RW      |    0.6976    |  0.6466 |
| SW      |    0.6730    |  0.8762 |

### MBERT + Cross-Switch (translating source language into random languages in chunk-level)

|languange  | Intent Acc.  | Slot F1 |
| --------- | ----------   | ------- |
| ES        |    0.9529    |  0.9638 |
| DE        |    0.9641    |  0.9685 |
| ZH        |    0.8902    |  0.9321 |
| JA        |    0.7872    |  0.8213 |
| PT        |    0.9406    |  0.9672 |
| FR        |    0.9630    |  0.9631 |
| HI        |    0.8488    |  0.8270 |
| TR        |    0.7930    |  0.8972 |
| RW        |    0.4524    |  0.8093 | 
| SW(random)|    0.6931    |  0.8461 |
| SW(to sw) |    0.9395    |  0.9117 |

*SW(random) means translating training set into random languages in chunk-level; SW(to sw) means translating training set into swahili in chunk-level.*







