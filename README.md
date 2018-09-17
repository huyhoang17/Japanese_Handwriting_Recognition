[![Build Status](https://travis-ci.org/huyhoang17/Japanese_Handwriting_Recognition.svg?branch=master)](https://travis-ci.org/huyhoang17/Japanese_Handwriting_Recognition)

# Japanese Handwriting Recognition

Datasets __(ETL1C)__
---

- Download ETL Dataset: http://etlcdb.db.aist.go.jp/?page_id=651
- Description: http://etlcdb.db.aist.go.jp/?page_id=1181

Command
---

```
export PYTHONPATH=path_to_Japanese_Recognition_project
```

Folder Structure
---

Root Folder
```
├── datasets
│   ├── ETL1
│   ├── ETL1C_data
│   └── sample
├── LICENSE
├── models
├── notebooks
├── papers
├── README.md
├── requirements.txt
├── src
│   └── utils.py
├── temp
└── web
```

ETL data (http://etlcdb.db.aist.go.jp/?page_id=651)
```
datasets/ETL1
├── ETL1C_01
├── ETL1C_02
├── ETL1C_03
├── ETL1C_04
├── ETL1C_05
├── ETL1C_06
├── ETL1C_07
├── ETL1C_08
├── ETL1C_09
├── ETL1C_10
├── ETL1C_11
├── ETL1C_12
├── ETL1C_13
└── ETL1INFO
```

ETL dataset
```
datasets/ETL1C_data/
├── 166
├── 168
├── 170
├── 177
├── 178
├── 179
├── 180
├── 181
├── 182
├── 183
...

datasets/ETL1C_data/200
├── 4233_1001_200_c8.png
├── 4234_1002_200_c8.png
├── 4235_1003_200_c8.png
├── 4236_1004_200_c8.png
├── 4237_1005_200_c8.png
├── 4238_1006_200_c8.png
├── 4239_1007_200_c8.png
├── 4240_1008_200_c8.png
├── 4241_1009_200_c8.png
├── 4242_1010_200_c8.png
...
```

Result
---

Reference
---

Custom data generator in keras
- https://medium.com/@ensembledme/writing-custom-keras-generators-fe815d992c5a
- https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
- https://keunwoochoi.wordpress.com/2017/08/24/tip-fit_generator-in-keras-how-to-parallelise-correctly