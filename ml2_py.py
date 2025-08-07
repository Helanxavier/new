import numpy as np
import pandas as pd
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv(r'C:\Users\X . HELAN SANTHIYA\Downloads\ML\dataset (1).csv')

df.isnull().sum()

df.columns

df.rename(columns = {"Nacionality": "Nationality",
                           "Mother's qualification": "Mother_qualification",
                           "Father's qualification": "Father_qualification",
                           "Mother's occupation": "Mother_occupation",
                           "Father's occupation": "Father_occupation",
                           "Age at enrollment": "Age"}, inplace = True)

df.columns = df.columns.str.replace(' ', '_')

col = ['Marital_status', 'Application_mode', 'Application_order', 'Course',
      'Daytime/evening_attendance', 'Previous_qualification', 'Nationality',
       'Mother_qualification', 'Father_qualification', 'Mother_occupation',
       'Father_occupation', 'Displaced', 'Educational_special_needs', 'Debtor',
       'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder',
      'International', 'Target']

df[col] = df[col].astype('category')

labels = df['Target'].value_counts().index
values = df['Target'].value_counts().values

plt.pie(values, labels = labels, colors = ['lightsalmon', 'skyblue', 'wheat'],
        autopct = '%1.0f%%')
plt.title('Proportion of the Labels')

from sklearn.preprocessing import OrdinalEncoder # Import OrdinalEncoder
df['Target_encoded'] = OrdinalEncoder(categories = [['Dropout', 'Enrolled', 'Graduate']]).fit_transform(df[['Target']])
df.drop('Target', axis = 1, inplace = True)

cats = ['Marital_status', 'Application_mode', 'Application_order',
        'Course','Daytime/evening_attendance', 'Previous_qualification',
        'Nationality','Mother_qualification', 'Father_qualification',
        'Mother_occupation', 'Father_occupation', 'Displaced',
        'Educational_special_needs', 'Debtor','Tuition_fees_up_to_date',
        'Gender', 'Scholarship_holder','International']

p_value = []

from scipy.stats import chi2_contingency

for col in cats:
    crosstable = pd.crosstab(index = df[col],
                             columns = df['Target_encoded'])
    p = chi2_contingency(crosstable)[1]
    p_value.append(p)

chi2_result = pd.DataFrame({
    'Variable': cats,
    'P_value': [round(ele, 5) for ele in p_value]
})

chi2_result = chi2_result.sort_values('P_value')

chi2_result

selected = df.drop(['Nationality', 'International', 'Educational_special_needs'],
                              axis = 1)

print(selected.columns)

columns_to_average = [
    ('Curricular_units_1st_sem_(credited)', 'Curricular_units_2nd_sem_(credited)'),
    ('Curricular_units_1st_sem_(enrolled)', 'Curricular_units_2nd_sem_(enrolled)'),
    ('Curricular_units_1st_sem_(evaluations)', 'Curricular_units_2nd_sem_(evaluations)'),
    ('Curricular_units_1st_sem_(approved)', 'Curricular_units_2nd_sem_(approved)'),
    ('Curricular_units_1st_sem_(grade)', 'Curricular_units_2nd_sem_(grade)'),
    ('Curricular_units_1st_sem_(without_evaluations)', 'Curricular_units_2nd_sem_(without_evaluations)')
]

for col1, col2 in columns_to_average:
    col1_clean = col1.split('(')[0].strip()
    col2_clean = col2.split('(')[0].strip()

    new_column_name = f'avg_{col1_clean}_{col2_clean}'
    selected[new_column_name] = selected[[col1, col2]].mean(axis=1)
print(selected.columns)

selected['avg_credited'] = selected[['Curricular_units_1st_sem_(credited)',
                                      'Curricular_units_2nd_sem_(credited)']].mean(axis = 1)
selected['avg_enrolled'] = selected[['Curricular_units_1st_sem_(enrolled)',
                                      'Curricular_units_2nd_sem_(enrolled)']].mean(axis = 1)
selected['avg_evaluations'] = selected[['Curricular_units_1st_sem_(evaluations)',
                                        'Curricular_units_2nd_sem_(evaluations)']].mean(axis = 1)
selected['avg_approved'] = selected[['Curricular_units_1st_sem_(approved)',
                                     'Curricular_units_2nd_sem_(approved)']].mean(axis = 1)
selected['avg_grade'] = selected[['Curricular_units_1st_sem_(grade)',
                                  'Curricular_units_2nd_sem_(grade)']].mean(axis = 1)
selected['avg_without_evaluations'] = selected[['Curricular_units_1st_sem_(without_evaluations)',
                                                 'Curricular_units_2nd_sem_(without_evaluations)']].mean(axis = 1)

num_features = selected[['Age', 'avg_credited', 'avg_enrolled',
                              'avg_evaluations', 'avg_approved',
                              'avg_grade', 'avg_without_evaluations',
                              'Unemployment_rate', 'Inflation_rate',
                              'GDP', 'Target_encoded']]

plt.figure(figsize = (12, 8))
plt.rcParams.update({'font.size': 8})
hm = sns.heatmap(num_features.corr(method = 'spearman'),
                 cmap = 'coolwarm', annot = True, fmt = '.2f',
                 linewidths = .2, vmin = -1, vmax = 1, center = 0)

"""Academic performance indicators (enrollment, approvals, grades) are strongly interconnected and are the most influential factors related to the target variable. Macroeconomic factors play a less direct role.  Potential multicollinearity exists among key academic variables."""

selected.loc[(selected['avg_approved'] == 0) & (selected['Target_encoded'] == 2)]

selected = selected.drop(selected.loc[(selected['avg_approved'] == 0) & (selected['Target_encoded'] == 2)].index)
selected.loc[(selected['avg_grade'] == 0) & (selected['Target_encoded'] == 2)]

columns_to_drop = ['Unemployment_rate', 'Inflation_rate',
                  'avg_credited', 'avg_evaluations',
                  'Curricular_units_1st_sem_credited',
                  'Curricular_units_1st_sem_enrolled',
                  'Curricular_units_1st_sem_evaluations',
                  'Curricular_units_1st_sem_approved',
                  'Curricular_units_1st_sem_grade',
                  'Curricular_units_1st_sem_without_evaluations',
                  'Curricular_units_2nd_sem_credited',
                  'Curricular_units_2nd_sem_enrolled',
                  'Curricular_units_2nd_sem_evaluations',
                  'Curricular_units_2nd_sem_approved',
                  'Curricular_units_2nd_sem_grade',
                  'Curricular_units_2nd_sem_without_evaluations']

existing_columns_to_drop = [col for col in columns_to_drop if col in selected.columns]
selected = selected.drop(columns=existing_columns_to_drop)

from sklearn.model_selection import train_test_split
train, test = train_test_split(selected, test_size = 0.2,
                               stratify = selected['Target_encoded'], random_state = 0)

train_features = train.drop('Target_encoded', axis = 1)
train_labels = train['Target_encoded']
test_features = test.drop('Target_encoded', axis = 1)
test_labels = test['Target_encoded']

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score # Import necessary metrics
rf_base = RandomForestClassifier(class_weight = 'balanced', random_state = 42)
rf_base.fit(train_features, train_labels)

y_pred = rf_base.predict(test_features)
y_prob = rf_base.predict_proba(test_features)

rf_base_accuracy = round(balanced_accuracy_score(test_labels, y_pred), 3)
rf_base_f1score = round(f1_score(test_labels, y_pred, average = 'macro'), 3)
rf_base_auc = round(roc_auc_score(test_labels, y_prob, average = 'macro', multi_class = 'ovr'), 3)

print('Random Forest Baseline Performance:')
print('Balanced Accuracy:', rf_base_accuracy)
print('F1 Score:', rf_base_f1score)
print('AUC score:', rf_base_auc)

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
parm = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 4, 5],
    'max_samples': [0.5, 0.75, 1]
}

# Tune the hyperparameters of the Random Forest
rsv_rf = RandomizedSearchCV(estimator = RandomForestClassifier(class_weight = 'balanced',
                                                               random_state = 42),
                            param_distributions = parm, scoring = 'balanced_accuracy',
                            n_iter = 30, n_jobs = -1,  random_state = 0)

rsv_rf.fit(train_features, train_labels)
tuned_rf = rsv_rf.best_estimator_
y_pred = tuned_rf.predict(test_features)
y_prob = tuned_rf.predict_proba(test_features)

tuned_rf_accuracy = round(balanced_accuracy_score(test_labels, y_pred), 3)
tuned_rf_f1score = round(f1_score(test_labels, y_pred, average = 'macro'), 3)
tuned_rf_auc = round(roc_auc_score(test_labels, y_prob, average = 'macro', multi_class = 'ovr'), 3)
print('Tuned Random Forest Performance:')
print('Balanced Accuracy:', tuned_rf_accuracy)
print('F1 Score:', tuned_rf_f1score)
print('AUC score:', tuned_rf_auc)

from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight(class_weight='balanced', y = train_labels)
xgb_base = XGBClassifier(enable_categorical = True, objective = 'multi:softmax',
                         num_class = 3, random_state = 42)
xgb_base.fit(train_features, train_labels, sample_weight=sample_weights)

y_pred = xgb_base.predict(test_features)
y_prob = xgb_base.predict_proba(test_features)

xgb_base_accuracy = round(balanced_accuracy_score(test_labels, y_pred), 3)
xgb_base_f1score = round(f1_score(test_labels, y_pred, average = 'macro'), 3)
xgb_base_auc = round(roc_auc_score(test_labels, y_prob, average = 'macro', multi_class = 'ovr'), 3)

print('XGBoost baseline performance:')
print('Balanced accuracy:', xgb_base_accuracy)
print('F1 score:', xgb_base_f1score)
print('AUC score:', xgb_base_auc)

import pandas as pd
performance = pd.DataFrame({
            'Model': ['rf_base', 'tuned_rf', 'xgb_base'],
            'Balanced Accuracy': [rf_base_accuracy, tuned_rf_accuracy, xgb_base_accuracy ],
            'F1 Score': [rf_base_f1score, tuned_rf_f1score, xgb_base_f1score],

            'AUC': [rf_base_auc, tuned_rf_auc, xgb_base_auc]
            })

fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (12, 4))
plt.subplots_adjust(wspace = 0.3)
x_ticks = range(len(performance['Model']))
models = performance['Model'].to_list()
axs[0].plot(x_ticks, performance['Balanced Accuracy'], linestyle = '-', color = 'lightgreen')
axs[0].set_title('Balanced Accuracy')
axs[0].set_xticks(x_ticks)
axs[0].set_xticklabels(models)
axs[0].set_ylim(0.68, 0.9)
y1 = performance['Balanced Accuracy'].to_list()
for i, y in enumerate(y1):
    axs[0].text(i, y+0.005, f'{y}', ha = 'center')

# Plot F1 Score
axs[1].plot(x_ticks, performance['F1 Score'], linestyle = '-', color = 'skyblue')
axs[1].set_title('F1 Score')
axs[1].set_xticks(x_ticks)
axs[1].set_xticklabels(models)
axs[1].set_ylim(0.68,0.9)
y2 = performance['F1 Score'].to_list()
for i, y in enumerate(y2):
    axs[1].text(i, y+0.005, f'{y}', ha = 'center')

# Plot AUC Score
axs[2].plot(x_ticks, performance['AUC'], linestyle = '-', color = 'lightsalmon')
axs[2].set_title('AUC')
axs[2].set_xticks(x_ticks)
axs[2].set_xticklabels(models)
axs[2].set_ylim(0.70, 0.91)
y3 = performance['AUC'].to_list()
for i, y in enumerate(y3):
    axs[2].text(i, y-0.01, f'{y}', ha = 'center')

"""XGBoost (xgb_base) is the top performer across all metrics. It has the highest Balanced Accuracy, F1 Score, and AUC, indicating superior overall performance.
Tuning improves the Random Forest model (tuned_rf). The tuned_rf model outperforms the base rf_base model in all metrics, demonstrating that hyperparameter optimization was effective.
The performance differences are most pronounced in the F1 Score. This suggests that the models' abilities to balance precision and recall vary significantly, and XGBoost excels in this aspect.
All models achieve relatively high AUC scores (above 0.9). This implies that all models are reasonably good at distinguishing between classes, though XGBoost still leads.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Calculate the confusion matrix
# Replacing X_test and y_test with the correct test features and labels: test_features and test_labels
cm = confusion_matrix(test_labels, xgb_base.predict(test_features), labels=xgb_base.classes_)

# Display the confusion matrix using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=xgb_base.classes_)
disp.plot(cmap='Blues')
plt.show()

"""The confusion matrix shows good overall performance, especially for class 2, but with some confusion between classes 0 and 1. The performance metrics plots confirm that XGBoost is the top performer, followed by the tuned Random Forest. The F1 score plot highlights the differences most strongly, indicating potential class imbalance or varying costs of misclassification."""

train_bi = train.drop(train[train['Target_encoded']==1].index)
test_bi = test.drop(test[test['Target_encoded']==1].index)

# Set the target label as 1 - 'Dropout', 0 - 'Graduate'
train_bi['Target_encoded'] = train_bi['Target_encoded'].replace([0, 2], [1, 0])
test_bi['Target_encoded'] = test_bi['Target_encoded'].replace([0, 2], [1, 0])

# Extract features and labels
train_bi_X = train_bi.drop('Target_encoded', axis = 1)
train_bi_y = train_bi['Target_encoded']
test_bi_X = test_bi.drop('Target_encoded', axis = 1)
test_bi_y = test_bi['Target_encoded']

rf_bi = RandomForestClassifier(class_weight = 'balanced', random_state = 42)
rf_bi.fit(train_bi_X, train_bi_y)

y_pred = rf_bi.predict(test_bi_X)
y_prob = rf_bi.predict_proba(test_bi_X)

rf_bi_accuracy = round(balanced_accuracy_score(test_bi_y, y_pred), 3)
rf_bi_f1score = round(f1_score(test_bi_y, y_pred), 3)
rf_bi_auc = round(roc_auc_score(test_bi_y, y_prob[:, 1]), 3)

print('Random Forest Baseline Performance:')
print('Balanced Accuracy:', rf_bi_accuracy)
print('F1 Score:', rf_bi_f1score)
print('AUC score:', rf_bi_auc)

neg_num = sum(train_bi_y == 0)
pos_num = sum(train_bi_y == 1)
weight = neg_num / pos_num

xgb_bi = XGBClassifier(enable_categorical = True, scale_pos_weight = weight,
                       importance_type = 'gain', random_state = 42)
xgb_bi.fit(train_bi_X, train_bi_y)

y_pred = xgb_bi.predict(test_bi_X)
y_prob = xgb_bi.predict_proba(test_bi_X)

xgb_bi_accuracy = round(balanced_accuracy_score(test_bi_y, y_pred), 3)
xgb_bi_f1score = round(f1_score(test_bi_y, y_pred), 3)
xgb_bi_auc = round(roc_auc_score(test_bi_y, y_prob[:, 1]), 3)

print('xgb_bi performance:')
print('Balanced accuracy:', xgb_bi_accuracy)
print('F1 score:', xgb_bi_f1score)
print('AUC score:', xgb_bi_auc)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

feature_importances = rf_bi.feature_importances_
feature_names = train_bi_X.columns
feature_imp = pd.DataFrame({
    'Features': feature_names,
    'Importance': feature_importances
})


feature_imp = feature_imp.sort_values(by='Importance', ascending=False)
feature_imp['cum_ratio'] = feature_imp['Importance'].cumsum() / feature_imp['Importance'].sum()

# Plotting
n = len(feature_imp)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

sns.barplot(data=feature_imp, x='Importance', y='Features', color='skyblue', ax=ax1)
ax1.set_title('Plot_1: Feature Importance Ranking')
ax2.plot(range(1, n + 1), feature_imp['cum_ratio'], color='skyblue')
ax2.plot(range(1, n + 1), np.repeat(0.95, n), color='grey', linestyle='dashed')
ax2.text(3, 0.96, '95%', color='grey', fontweight='bold')
ax2.set_title('Plot_2: Cumulative Sum of Importance Ratio')
ax2.set_xlabel('Number of Features')
ax2.set_ylabel('CumSum of Importance Ratio')
ax2.set_xticks(range(1, len(feature_imp) + 1), range(1, len(feature_imp) + 1))

plt.tight_layout()
plt.show()

least_imp = feature_imp['Features'][-5:].to_list()
train_16X = train_bi_X.drop(columns = least_imp)
test_16X = test_bi_X.drop(columns = least_imp)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, classification_report


rf_b16 = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_b16.fit(train_16X, train_bi_y)
y_pred = rf_b16.predict(test_16X)
y_prob = rf_b16.predict_proba(test_16X)

rf_b16_accuracy = round(balanced_accuracy_score(test_bi_y, y_pred), 3)
rf_b16_f1score = round(f1_score(test_bi_y, y_pred), 3)
rf_b16_auc = round(roc_auc_score(test_bi_y, y_prob[:, 1]), 3)
rf_b16_precision = round(precision_score(test_bi_y, y_pred), 3)
rf_b16_recall = round(recall_score(test_bi_y, y_pred), 3)
rf_b16_confusion_matrix = confusion_matrix(test_bi_y, y_pred)

print('rf_b16 Performance:')
print('Balanced Accuracy:', rf_b16_accuracy)
print('F1 Score:', rf_b16_f1score)
print('AUC score:', rf_b16_auc)
print('Precision:', rf_b16_precision)
print('Recall:', rf_b16_recall)
print('Confusion Matrix:\n', rf_b16_confusion_matrix)

print('\nClassification Report:\n', classification_report(test_bi_y, y_pred))

from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

neg_num = sum(train_bi_y == 0)
pos_num = sum(train_bi_y == 1)
weight = neg_num / pos_num

xgb_b16 = XGBClassifier(enable_categorical=True, scale_pos_weight=weight,
                        importance_type='gain', random_state=48)
xgb_b16.fit(train_16X, train_bi_y)
y_pred = xgb_b16.predict(test_16X)
y_prob = xgb_b16.predict_proba(test_16X)

xgb_b16_accuracy = round(balanced_accuracy_score(test_bi_y, y_pred), 3)
xgb_b16_f1score = round(f1_score(test_bi_y, y_pred), 3)
xgb_b16_auc = round(roc_auc_score(test_bi_y, y_prob[:, 1]), 3)
xgb_b16_precision = round(precision_score(test_bi_y, y_pred), 3)
xgb_b16_recall = round(recall_score(test_bi_y, y_pred), 3)
xgb_b16_confusion_matrix = confusion_matrix(test_bi_y, y_pred)

print('xgb_b16 performance:')
print('Balanced accuracy:', xgb_b16_accuracy)
print('F1 score:', xgb_b16_f1score)
print('AUC score:', xgb_b16_auc)
print('Precision:', xgb_b16_precision)
print('Recall:', xgb_b16_recall)
print('Confusion Matrix:\n', xgb_b16_confusion_matrix)


print('\nClassification Report:\n', classification_report(test_bi_y, y_pred))

xgb_b16 = XGBClassifier(enable_categorical = True, scale_pos_weight = 1.65,
                        importance_type = 'gain', random_state = 48)
xgb_b16.fit(train_16X, train_bi_y)

y_pred = xgb_b16.predict(test_16X)
y_prob = xgb_b16.predict_proba(test_16X)

xgb_b16_accuracy = round(balanced_accuracy_score(test_bi_y, y_pred), 3)
xgb_b16_f1score = round(f1_score(test_bi_y, y_pred), 3)
xgb_b16_auc = round(roc_auc_score(test_bi_y, y_prob[:, 1]), 3)

print('xgb_b16 performance:')
print('Balanced accuracy:', xgb_b16_accuracy)
print('F1 score:', xgb_b16_f1score)
print('AUC score:', xgb_b16_auc)

import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


warnings.simplefilter(action='ignore', category=FutureWarning)
results = pd.DataFrame({
    'Models': ['rf_bi', 'xgb_bi', 'rf_bi', 'xgb_bi', 'rf_bi', 'xgb_bi'],
    'Metrics': ['Balanced Accuracy', 'Balanced Accuracy', 'F1 Score', 'F1 Score', 'AUC', 'AUC'],
    'Performance': [rf_bi_accuracy, xgb_bi_accuracy, rf_bi_f1score,
                    xgb_bi_f1score, rf_bi_auc, xgb_bi_auc]
})

assert len(results['Models']) == len(results['Metrics']) == len(results['Performance']), "Column lengths do not match"
plt.figure(figsize=(6, 6))
xticks = range(len(results['Models'].unique()))
mods = ['First Iteration \n rf_bi', 'Second Iteration \n xgb_bi']

ax = sns.lineplot(data=results, x='Models', y='Performance', hue='Metrics',
                  palette=['skyblue', 'lightsalmon', 'lightgreen'], linewidth=2)

ax.set_xticks(xticks)
ax.set_xticklabels(mods)
ax.set_xlabel('')
for x, y in zip([0, 1] * 3, results['Performance']):
    ax.text(x, y + 0.002, f'{y: .2f}', ha='center')
ax.set_title('Performance Improvement of Models Over Two Iterations')
ax.legend(title='', loc='lower right')

plt.tight_layout()
plt.show()

"""XGBoost Improvement: XGBoost demonstrates a noticeable increase in all three metrics, particularly Balanced Accuracy and F1 Score, moving from the first to the second iteration.
Random Forest Stagnation: Random Forest shows minimal improvement, with only a slight increase in AUC and virtually no change in Balanced Accuracy and F1 Score.
In short: While both models started at similar performance levels in the first iteration, XGBoost significantly outperformed Random Forest in the second iteration, indicating a greater capacity for improvement with further development or tuning.
"""

important_features = train_16X.columns.tolist()
def user_input_prediction(model):
    user_data = {}
    for feature in important_features:
        user_data[feature] = float(input(f"Enter value for {feature}: "))

    user_df = pd.DataFrame([user_data])
    prediction = model.predict(user_df)
    print(f"Prediction: {prediction}")


rf_b16.fit(train_16X, train_bi_y)
xgb_b16.fit(train_16X, train_bi_y)

model_choice = input("Enter the model you want to use (rf_b16 / xgb_b16): ").strip()

if model_choice == 'rf_b16':
    user_input_prediction(rf_b16)
elif model_choice == 'xgb_b16':
    user_input_prediction(xgb_b16)
else:
    print("Invalid model choice. Please enter 'rf_b16' or 'xgb_b16'.")

import pickle
pickle.dump(xgb_b16, open('dropped.sav', 'wb'))

import pickle
pickle.dump(xgb_b16, open('dropped.pkl', 'wb'))