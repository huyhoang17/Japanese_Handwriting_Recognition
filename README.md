# Japanese Handwriting Recognition

Datasets
---

- Download ETL Dataset: http://etlcdb.db.aist.go.jp/?page_id=651
- Description: http://etlcdb.db.aist.go.jp/?page_id=1181

Command
---

```
python3 src/utils.py
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
├── =
├── -
├── ,
├── '
├── (
├── )
├── *
├── +
├── ツ
├── エ
├── ハ
├── ヲ
├── ウ
├── ソ
├── ヒ
├── ネ
├── タ
├── ン
├── ロ
...
datasets/ETL1C_data/A
├── 2891-1001-65-c1.png
├── 2892-1002-65-c1.png
├── 2893-1003-65-c1.png
├── 2894-1004-65-c1.png
├── 2895-1005-65-c1.png
├── 2896-1006-65-c1.png
├── 2897-1007-65-c1.png
├── 2898-1008-65-c1.png
├── 2899-1009-65-c1.png
├── 2900-1010-65-c1.png
...
```

Reference
---