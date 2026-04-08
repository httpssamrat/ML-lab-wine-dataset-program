# 🍷 Wine Classification ML Lab

A complete machine learning lab implementation based on the UCI Wine Dataset.

## 📌 Features
- Data preprocessing & cleaning
- Visualization (Seaborn + Matplotlib)
- Regression & Classification
- Ensemble Learning
- Clustering
- ANN & CNN
- NLP preprocessing
- Generative AI experiments
- vUCI Wine Recognition Dataset
178 samples, 13 features, 3 classes
---

## ⚙️ Setup

```bash
git clone https://github.com/httpssamrat/wine-ml-lab.git
cd wine-ml-lab
bash setup.sh
# Data Loading
pd.read_csv()

# Data Cleaning
df.isnull()
df.fillna()
df.astype()
df.rename()

# Visualization
sns.pairplot()
plt.hist()
plt.scatter()
plt.boxplot()

# ML
train_test_split()
StandardScaler()
fit()
predict()

# Metrics
accuracy_score()
confusion_matrix()
classification_report()

# Hyperparameter tuning
GridSearchCV()

# Clustering
KMeans()
AgglomerativeClustering()

# NLP
tokenize()
stopwords()
stemming()
lemmatization()
TfidfVectorizer()

# Deep Learning
MLPClassifier()
TensorFlow CNN
