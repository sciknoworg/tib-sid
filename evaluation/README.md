## 📊 Quantitative Evaluation

The following metrics are used for assessment:

1. **Average Precision@k** for k = 5, 10, 15, 20
2. **Average Recall@k** for k = 5, 10, 15, 20
3. **Average F1-score@k** for k = 5, 10, 15, 20
4. **nCDG@k** for k = 5, 10, 15, 20

Evaluation results are offered at varying levels of granularity to provide comprehensive insights:

1. **Language-level**: Separate evaluations for English and German.
2. **Record-level**: Evaluations for each of the five types of technical records.
3. **Combined Language and Record-levels**: Detailed evaluations combining both language and record type.

## 🛠️ Evaluation Script

Run the [script](llms4subjects-evaluation.py).

### Execution Instructions

**🗂️ Step 1: Preparing the Folder Structure**

The script requires a specific folder structure, identical to the train and dev sets, which are organized by record type and language.

Predictions should be stored in a JSON file, named identically to the corresponding record file, containing the predicted subject tags as a list of GND IDs.

**▶️ Step 2: Running the Script**

The script requires three user inputs:

1. The path to the gold-standard dataset with the annotations.
2. The path to the model's predictions.
3. The path to save the results as an Excel file.

The script will generate an Excel file containing the evaluation metrics scores, organized into three different sheets, each corresponding to a different level of granularity.

### 🧑‍💻 Code Execution Sample

```bash
$python llms4subjects-evaluation.py

Please enter your Team Name
Team Name> test

Please specify the directory containing the true GND labels
Directory path> evaluation/all_subjects

Please specify the directory containing the predicted GND labels
Directory path> evaluation/all_subjects/run1

Please specify the directory to save the evaluation metrics
Directory path> evaluation/results

Reading the True GND labels...
Reading the Predicted GND labels...

Evaluating the predicted GND labels...

File containing the evaluation metrics score is saved at location: evaluation/results/test_evaluation_metrics.xlsx
```
