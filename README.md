# Term Project - Pankaj Yawale
Mushroom classification

## Citation
Mushroom [Dataset]. (1981). UCI Machine Learning Repository. https://doi.org/10.24432/C5959T.

## Download Code
```bash
git clone https://github.com/MET-CS-777/cs777-final-project-ypankaj30.git
```

Alternatively, you can download the code as a zip file and unzip it in a folder.

From the folder where the repo is cloned, or the zip file is unzipped, run the following command.

```bash
cd cs777-final-project-ypankaj30/
```
## Run Classifier

### Default
If no arguments are provided, it uses the Random Forest classifier with 10 trees and all (no feature selection) features

```bash
python3 Yawale_Pankaj_Term_Project.py
```

### Arguments
Supports two arguments for specifying the classifier and number of features to be selected using ChiSqSelector

First argument can be either "rf" for Random Forest, "dt" for Decision Tree or "nb" for Naive Bayes.

Arguments for Random Forest and Decision Tree can be suffixed with a ":" separator too specify the respective hyper parameters. e.g. Number of trees for Random Forest and max depth for Decision Tree.

Second argument can be any number smaller than 22 (total features).

For example, to train the Random Forest classifier with 10 trees and 12 features - 

```bash
python3 Yawale_Pankaj_Term_Project.py rf:10 12
```

### Random Forest Classifier
For, 5 trees and 20 features

```bash
python3 Yawale_Pankaj_Term_Project.py rf:5 20
```

### Decision Tree Classifier
For, max depth as 8 and 15 features

```bash
python3 Yawale_Pankaj_Term_Project.py dt:8 15
```

### Naive Bayes Classifier
For, 12 features

```bash
python3 Yawale_Pankaj_Term_Project.py nb 12
```
