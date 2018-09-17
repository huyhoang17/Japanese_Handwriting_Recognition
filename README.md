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
datasets/ETL1C_data
├── 20
├── 27
├── 28
├── 29
├── 2a
├── 2b
├── 2c
├── 2d
├── 2e
├── 2f
...

datasets/ETL1C_data/2a
├── 4336_1001_2a.png
├── 4337_1002_2a.png
├── 4338_1003_2a.png
├── 4339_1004_2a.png
├── 4340_1005_2a.png
├── 4341_1006_2a.png
├── 4342_1007_2a.png
├── 4343_1008_2a.png
├── 4344_1009_2a.png
├── 4345_1010_2a.png
...
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