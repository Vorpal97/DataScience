import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
class LinearDiscriminant:
    pass

class XGBClassifier:
    pass

import pandas as pd
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


def classification(df):
    Type = df.type_1.unique()
    print(Type)
    df['type_1'] = df['type_1'].rank(method='dense').astype(int)
    x = df.values
    y = df.type_1

    rs = 42

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=rs)

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    classifiers = [
        LogisticRegression(random_state= rs),
        DecisionTreeClassifier(random_state= rs),
        RandomForestClassifier(n_estimators= 10, random_state=rs),
        MLPClassifier()
    ]

    clf_name = []
    model_results = pd.DataFrame.copy(y_test)

    kfold = StratifiedKFold(n_splits=5)
    cv_results = []
    cv_acc = []
    cv_std = []

    cnfm = []
    clr = []

    for clf in classifiers:
        name = clf.__class__.__name__
        clf_name.append(name)

        model = clf.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        model_results[name] = y_pred

        cv_results.append(cross_val_score(clf, x_train, y_train, scoring="accuracy", cv=kfold))
        acc = round(accuracy_score(y_test, y_pred), 2)
        train_pred = clf.predict_proba(x_test)
        print(f'Accuracy: {acc}\t ---->  {name} ')

        cnfm.append(confusion_matrix(y_test, y_pred))
        clr.append(classification_report(y_test, y_pred))

    for i in cv_results:
        cv_acc.append(i.mean())
        cv_std.append(i.std())

    cv_res = pd.DataFrame({"CrossValMeans": cv_acc, "CrossValerrors": cv_std, "Algorithm": clf_name})
#confronto classificatori
    plt.figure(figsize=(12,6))
    sns.barplot("CrossValMeans", "Algorithm", data = cv_res, palette="Set2", orient = 'h', **{'xerr':cv_std})
    plt.xlabel("Mean Accuracy")
    plt.title("Cross Validation scores")
    plt.savefig("grafici/barplot.png", dpi=356, bbox_inches='tight')
    plt.show()
#heatmap
    for i in range(len(classifiers)):
        hm = sns.heatmap(cnfm[i], annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(clf_name[i])
        plt.show()
        figure = hm.get_figure()
        figure.savefig("grafici/HM" + str(i) + ".png", dpi=356, bbox_inches='tight')
#report
    for i in range(len(classifiers)):
        print(f"{clf_name[i]} Classification Report:" );
        print (clr[i]);