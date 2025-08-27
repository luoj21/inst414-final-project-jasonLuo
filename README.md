## Patient DCIS Risk Classification Using Machine Learning
__INST414 - Data Science Techniques Final Project__


-----

**Business Problem**:

Current detection tools for Ductal carcinoma in situ (DCIS) include mammograms that indicate whether or not a patient has DCIS, but there is a lack of tools that can correctly distinguish between low and high-risk DCIS. As a result, many patients may receive too much or too little treatment, leading to unstable healthcare costs and increased patient anxiety. This project attempts to solve this problem using publicly available clinical data to accurately classify low, medium, and high-risk DCIS patients.

-----

**To get started**:
- Do git clone ```https://github.com/luoj21/inst414-final-project-jasonLuo.git```
- Create a virtual envrionment: ```python3 -m venv .venv```
- Activate the virtual envrionment: ```source .venv/bin/activate```
- Then do ```pip install -r requirements.txt```
- In the project folder run ```python main.py```


-----

**Code Package Structure**

``` bash
├── analysis
│   ├── evaluate_model.py
│   └── models.py
├── data
│   ├── outputs
│   │   ├── age_hist_plot.png
│   │   ├── class_counts_plot.png
│   │   ├── classification_report.csv
│   │   ├── Confusion Matrix For FOREST.png
│   │   ├── Confusion Matrix For KNN.png
│   │   ├── Confusion Matrix For LDA.png
│   │   ├── days_last_follow_up_hist_plot.png
│   │   ├── ethnicity_counts_plot.png
│   │   ├── multiclass_roc.png
│   │   └── tumors_by_age.png
│   ├── raw_data
│   │   ├── demographics.csv
│   │   ├── diagnostics.csv
│   │   └── molecular_test.csv
│   ├── reference-tables
│   │   ├── data_dictionary_demographics.csv
│   │   ├── data_dictionary_diagnostics.csv
│   │   ├── data_dictionary_merged_df.csv
│   │   └── data_dictionary_molecular_test.csv
│   └── transformed_data
│       └── merged_df.csv
├── etl
│   ├── extract.py
│   └── transform.py
├── logger_config.py
├── logs
│   └── LOGS.log
├── main.py
├── README.md
├── requirements.txt
└── vis
    └── visualizations.py
```

-----