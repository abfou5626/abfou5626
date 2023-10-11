import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import pair_confusion_matrix, precision_recall_curve, precision_score, recall_score
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_fscore_support
from sklearn.datasets import make_regression
# from sklearn.metrics import plot_roc_cur



def main():
    st.title("Application de machine learning pour la dection des fraudes bancaire")
    st.subheader("Abdou Garba Hamisssou")

    # data = make_regression(n_features=1)
    # hist_values = np.histogram(data[0], bins=24, range=(0, 24))[0]

   # st.bar_chart(hist_values)
    @st.cache_data(persist=True)  
    def chargedata():
        data=pd.read_csv("creditcard.csv")
        return data
    #affichage des donnees

    df=chargedata()
    df_sample=df.sample(100)
    if st.sidebar.checkbox("Afficher les données brutes", False):
        st.subheader("jeu des données de 100 echantillons")
        st.write(df_sample)
    
    seed=123
    
    
    y=df['Class'] 
    x= df.drop('Class',axis=1)
    x_train,x_test,y_train,y_test=train_test_split(x, y,test_size=0.25,
        stratify=y)
         
    

    

    classifier=st.sidebar.selectbox(
        "Classificateur",("Random Forest","SVM","Support vector machine")
    )

    #analyse de performance de notre model
    def plot_perfo(grapes):
         if 'Confusion matrix' in grapes: 
              st.subheader("matrice de confusion")
              pair_confusion_matrix(x_test,y_test)
              st.pyplot()
        #  if 'ROC curve' in grapes: 
        #     st.subheader("courve roc")
        #     plot_roc_curve(model, x_test, y_test)

        #     st.pyplot()
         if 'Precision recall' in grapes: 
            st.subheader("presion rappel")
            precision_recall_curve(x_test,y_test)

            st.pyplot()
    # random forest
    if classifier=="Random Forest":
        st.sidebar.subheader("hyperparametre du model")
        n_arbre=st.sidebar.number_input("choisir le nombre d'arbre de la foret",
         100,1000,step=10)
        profondeur_arbre=st.sidebar.number_input("profondeur d'un arbre",1,20 ,step=1)
        bootstrap=st.sidebar.radio("Echontillons bootstrap  lors de creation de l'arbre",
                                   (1,0))
        graphes_perf=st.sidebar.multiselect(
            "choisir un graphe de performances du model",
            ("Confusion matrix","ROC curve","Precision recall")
        )
        if st.sidebar.button("Execution",key="classify"):
            st.subheader("Random forest results")
            #initialisation d'un objet random forest
            model=RandomForestClassifier(n_estimators=n_arbre, max_depth=profondeur_arbre,
                                         bootstrap=bootstrap)
            # Entrainement d'un model
            model.fit(x_train,y_train)

            #prediction
            y_pred=model.predict(x_test)
            #calcul de metric de performances
            accuration=model.score(x_test, y_test)
            precision=precision_score(y_test,y_pred)
            recall=recall_score(y_test,y_pred,average='macro'  )
            st.write("Accuration:",accuration)
            st.write("Precision:",precision)
            st.write("Recall:",recall)

        plot_perfo(graphes_perf)



main()
