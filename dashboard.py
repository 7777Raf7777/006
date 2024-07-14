import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import load_model
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import recall_score, f1_score

def load_data(file, nrows):
    data= pd.read_csv(file)
    return data.head(nrows)

dataX = load_data("/content/drive/MyDrive/Projet Dreem/X_train.csv",50)
dataY = load_data("/content/drive/MyDrive/Projet Dreem/y_train.csv",947)
dataY = dataY.drop('id', axis=1)


def get_model_summary(model):
    from io import StringIO
    import sys

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    model.summary()
    summary_string = sys.stdout.getvalue()
    sys.stdout = old_stdout
    return summary_string

if 'label' in dataY.columns:
    
    dataY['gender_label'] = dataY['label'].map({0: 'Female', 1: 'Male'})
    g_count = dataY['gender_label'].value_counts()
    g_count_df = g_count.reset_index()
    g_count_df.columns = ['gender_label', 'count'] 

    
    fig_pie = px.pie(g_count_df, names='gender_label', values='count',
                     title='Distribution du Dataset', hole=0.4)


with h5py.File("/content/drive/MyDrive/Projet Dreem/X_test_new.h5", 'r') as f:
    ls = list(f.keys())
    data = f.get('features')
    dataset = np.array(data)

modelcnn = load_model('/content/drive/MyDrive/Projet Dreem/CNN_model.h5')

def predict_sexe(dataset):
    predict = modelcnn.predict(dataset)
    return np.argmax(predict, axis=1) 

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Homme', 'Femme'], yticklabels=['Homme', 'Femme'], cbar=False, ax=ax)
    ax.set_xlabel('Prédictions')
    ax.set_ylabel('Vraies étiquettes')
    ax.set_title('Matrice de Confusion')
    return fig


def main():
    st.title('Prédire le sexe à partir de son activité cérébrale')

    option = st.sidebar.selectbox('Choose the data view:', ('X Data', 'Y Data', 'Distribution Femme/Homme'))

    if option == 'X Data':
        st.write("X Data", dataX)
        st.write(" ")  
        st.write(" ")

    elif option == 'Y Data':
        st.write("Y Data", dataY)
        st.write(" ")  
        st.write(" ")

    elif option == 'Distribution Femme/Homme':
        st.plotly_chart(fig_pie)
        st.write(" ")  
        st.write(" ")

    if st.sidebar.button('View Signal Plots'):
        t = 2
        fr = 250
        x = [t / fr for t in range(len(dataset[0][0][0]))]
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 8))
        titles = ['sig0', 'sig1', 'sig2', 'sig3', 'sig4', 'sig5', 'sig6']

        for i in range(7):
            row, col = divmod(i, 4)
            sig = dataset[0][0][i]
            axes[row, col].plot(x, sig)
            axes[row, col].set_title(titles[i])

        plt.tight_layout(pad=3.0)
        fig.suptitle('Les différents signaux', fontsize=15, y=0.95)

        if axes.shape[1] == 4:
            axes[1, 3].axis('off')

        st.pyplot(fig)
        
    cnn_model_option = st.sidebar.selectbox(
        'CNN Model Actions:',
        ('Sommaire du Modèle', 'Prediction du Modèle CNN', 'Matrice de Confusion')
    )
    if cnn_model_option == 'Sommaire du Modèle':
        summary = get_model_summary(modelcnn)
        st.text(summary)

    elif cnn_model_option == 'Prediction du Modèle CNN':
        num_predictions = st.number_input('How many predictions?', min_value=1, max_value=len(dataset), value=5)
        selected_data = dataset[:num_predictions] 

        predictions = predict_sexe(selected_data)
        results = ['Homme' if predict == 0 else 'Femme' for predict in predictions]

        results_counter = Counter(results)

        for i, result in enumerate(results):
            st.write(f"Prediction {i + 1}: {result}")

        st.write("Nombre de prédictions 'Homme':", results_counter['Homme'])
        st.write("Nombre de prédictions 'Femme':", results_counter['Femme'])

    elif cnn_model_option == 'Matrice de Confusion':  
      y_true = dataY['label'].values  
      predictions = predict_sexe(dataset)

      cm = confusion_matrix(y_true, predictions)
      fig = plot_confusion_matrix(cm)
      st.pyplot(fig) 

      plt.close(fig)

      recall = recall_score(y_true, predictions)
      f1 = f1_score(y_true, predictions)

      st.write(f"Recall: {recall:.2f}")
      st.write(f"F1 Score: {f1:.2f}")

if __name__ == '__main__':
    main()
