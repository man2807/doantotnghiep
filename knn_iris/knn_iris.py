import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

def data_split(data):
    data_map = {'setosa': 1,'versicolor': 2, 'virginica': 3} #convert data lable
    data['Species'] = data['Species'].map(data_map) 
    X = data.values[:,:3] 
    y = data.values[:,4]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    return X_train, X_test, y_train, y_test

def pyplot(data,accu_knn, conf_matrix_knn,conf_matrix_knn_tree, accu_tree):
    plt.figure(figsize=(4,4))
    plt.title('So sánh dữ liệu nhãn giữa các lớp')
    sns.countplot(x =  data['Species'],order = data['Species'].value_counts().index)
    plt.figure(figsize=(4,4))
    plt.title(f'Accuracy: {accu_knn*100:0.2f}% \nConfusion Matrix KNN')
    df_cm_knn = pd.DataFrame(conf_matrix_knn)
    sns.heatmap(df_cm_knn, annot=True, annot_kws={'size': 12}, cmap='crest', linewidths=0.1, linecolor='black')
    plt.figure(figsize=(4,4))
    plt.title(f'Accuracy: {accu_tree*100:0.2f}% \nConfusion Matrix TREE')
    df_cm_tree = pd.DataFrame(conf_matrix_knn_tree)
    sns.heatmap(df_cm_tree, annot=True, annot_kws={'size': 12}, cmap='crest', linewidths=0.1, linecolor='black')
    plt.show()

def predict_knn(k,X_train,y_train,X_test):
    knn = KNN(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    return y_pred_knn

def predict_tree(X_train,y_train,X_test):
    clf = DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def main():
    #load data
    data = pd.read_csv("./iris.csv") # read data
    print(data.head())

    X_train, X_test, y_train, y_test = data_split(data)
    
    #KNN
    y_pred_knn = predict_knn(3, X_train, y_train, X_test) # k = 3
    conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
    accu_knn = accuracy_score(y_test, y_pred_knn)
    print('Test Accuracy KNN: ',accu_knn*100)
    print('confusion_matrix KNN: \n', conf_matrix_knn)

    #Decisicon_tree
    y_pred_tree = predict_tree(X_train, y_train, X_test)
    conf_matrix_tree = confusion_matrix(y_test, y_pred_tree)
    accu_tree = accuracy_score(y_test, y_pred_tree)
    print('\nTest Accuracy Tree: ',accu_tree*100)
    print('confusion_matrix Tree: \n', conf_matrix_tree)

    #Draw
    pyplot(data,accu_knn, conf_matrix_knn,conf_matrix_tree,accu_tree)

if __name__ == '__main__':
    main()