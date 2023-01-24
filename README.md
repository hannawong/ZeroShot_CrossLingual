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

#### BERT (bert-base-uncased)

Intent Acc.: PT: 0.3314669652855543
Slot F1: PT: 0.8217050489313014
Intent Acc.: FR: 0.4154535274356103
Slot F1: FR: 70548712206
Intent Acc.: HI: 0.032474804031354984
Slot F1: HI: 0.35047950.830907544184233
Intent Acc.: TR: 0.15804195804195803
Slot F1: TR: 0.7090605047126787

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


