# Cor Jesu College Engineering Data Analysis

Study habits and one-way commute times for 32 undergraduate respondents, prepared by **Joseph Karl V. Boncavil, BSCoE, 3rd Year** for Engineering Data Analysis, Chapters 1-3.

The responses were collected in the lobby of Norbert Building on 14 October 2026. Available and willing students were approached on an examination day, so the report describes convenience sampling and the resulting limits on generalization.

## Open the results

- [Executed Jupyter notebook](Cor_Jesu_EDA_Analysis.ipynb)
- [A4 PDF report with personalized cover](analysis_output/Cor_Jesu_EDA_Report.pdf) - one cover page and three analysis pages
- [Charts and frequency/statistics tables](analysis_output/)
- [Anonymous survey responses](survey_responses.csv)

The sample mean study time is **2.34 hours per typical school day** and the sample mean one-way commute is **25.16 minutes**. Both the 0-minute and 55-minute commute observations are retained; an IQR flag is not an automatic exclusion.

## Run the notebook

Use Python 3.10 or newer. From the repository folder, install the dependencies and open JupyterLab:

```sh
python -m pip install -r requirements.txt
jupyter lab Cor_Jesu_EDA_Analysis.ipynb
```

Select a Python kernel, then choose **Restart Kernel and Run All Cells**. Run the notebook from the repository folder so its relative CSV paths resolve correctly. It regenerates the tables, six chart images, and PDF in [analysis_output/](analysis_output/). The supplied reference PDFs are not required to execute it.

Student, course, collection, and cover information can be edited in the notebook's `METADATA` and `COVER_DETAILS` dictionaries.

## Files and calculation conventions

| File | Purpose |
| --- | --- |
| [survey_responses.csv](survey_responses.csv) | Primary dataset: one anonymous respondent per row |
| [survey_dataset.csv](survey_dataset.csv) | Identical copy of the current primary dataset |
| [data_dictionary.csv](data_dictionary.csv) | Variable definitions, measurement levels, units, and permitted values |
| [collection_log.csv](collection_log.csv) | Known collection details; undocumented selection order remains blank |
| [survey_analysis.py](survey_analysis.py) | Validation, tables, descriptive statistics, outlier flags, and charts |
| [survey_report.py](survey_report.py) | Cover and report layout |

The notebook embeds the two Python modules so its analysis code is visible and self-contained. After changing a module, update the corresponding notebook code cell as well.

Variance and standard deviation use the sample denominator (`ddof=1`). Quartiles and percentiles use NumPy's inclusive linear interpolation, matching the exam's spreadsheet INC convention; this can differ from Bluman's hand-calculation rules. Boxplots use the modified Tukey convention with 1.5-times-IQR fences. No open-ended study-hour answer occurs in the current dataset.

## References

Bluman, A. G. (2012). *Elementary Statistics: A Step by Step Approach* (8th ed.). McGraw-Hill, sections 1-1 to 1-3, 2-1 to 2-3, and 3-1 to 3-4.

The Cor Jesu College applied-exam instructions and printed survey form define the assignment and questionnaire. These instructor-supplied PDFs and the textbook remain local reference materials and are not distributed in this repository.
