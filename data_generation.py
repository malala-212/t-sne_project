import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from pathlib import Path
import os

from sklearn.datasets import make_blobs


def create_directory(n_components, project_path = './data/'):
    """
    Crée un répertoire pour le projet et un sous-répertoire pour chaque composant (cluster).

    Cette fonction vérifie si le répertoire de base existe déjà. Si ce n'est pas le cas, il est créé. Ensuite,
    un sous-répertoire spécifique au nombre de composants est créé à l'intérieur du répertoire de base.

    Arguments :
        n_components (int) : Le nombre de composants (ou de clusters) pour lesquels un sous-répertoire sera créé.
        project_path (str, optional) : Le répertoire de base où les données seront stockées. Par défaut, './data/'.

    Retour :
        str : Le chemin du sous-répertoire créé, où les données liées à "n_components" seront stockées.
    """
   
    if not os.path.exists(project_path):
        os.mkdir(project_path)
        print(f'Directory "{project_path}" created')

    job_path = f'center{n_components}'
    output = os.path.join(project_path,job_path)
    if not os.path.exists(output):
        os.mkdir(output)
        print(f'Directory "{output}" created')

    return output

def create_data(n_samples,feat,n_components):
    """
    Génère un jeu de données synthétiques à l'aide de "make_blobs", avec des échantillons et des caractéristiques
    spécifiées, et associe chaque échantillon à un cluster.

    Cette fonction utilise "make_blobs" de "sklearn" pour générer des données en 2D (ou plus si spécifié). Elle 
    retourne un DataFrame avec des échantillons, leurs caractéristiques et un label de cluster.

    Arguments :
        n_samples (int) : Le nombre d'échantillons à générer.
        feat (int) : Le nombre de caractéristiques (dimensions) pour chaque échantillon.
        n_components (int) : Le nombre de clusters (centres) pour les données générées.

    Retour :
        pandas.DataFrame : Un DataFrame contenant les caractéristiques ("Feature_1", "Feature_2", ...) et un label 
                            de cluster pour chaque échantillon ("Cluster_Label").
    """

    x, y = make_blobs(n_samples=n_samples, n_features=feat, centers=n_components)

    # print("X: ",x)
    # print("y: ",y)

    feature_cols = [f'Feature_{i+1}' for i in range(feat)]

    df_X = pd.DataFrame(x, columns=feature_cols)


    df_X['Cluster_Label'] = y

    return df_X

def save_data (feat, df_X,output):
    """
    Sauvegarde les données générées dans un fichier CSV dans le répertoire spécifié.

    Cette fonction prend le DataFrame "df_X" contenant les caractéristiques et le label de cluster, et le sauvegarde
    dans un fichier CSV. Le nom du fichier inclut le nombre de caractéristiques utilisées pour la génération des 
    données.

    Arguments :
        feat (int) : Le nombre de caractéristiques utilisées dans la génération des données.
        df_X (pandas.DataFrame) : Le DataFrame contenant les données à sauvegarder (incluant les étiquettes de clusters).
        output (str or Path) : Le répertoire où le fichier CSV sera sauvegardé.

    Retour :
        None
    """

    try:   
        file_name = f"features{feat}.csv"
        df_X.to_csv(os.path.join(output,file_name), index=False) # index=False pour ne pas sauvegarder l'index du DataFrame

    except Exception as e :
        print("Error : ",e)

    else :
        print("Data succefully saved.")

    return


def load_data(features,root_dir = "data"):
    """
    Charge les fichiers CSV contenant des données générées et renvoie les caractéristiques et les labels des clusters.

    Cette fonction parcourt tous les fichiers CSV présents dans le répertoire racine (et ses sous-répertoires), 
    charge chaque fichier, extrait les caractéristiques et les labels des clusters, et les retourne sous forme 
    de listes.

    Arguments :
        features (int) : Le nombre de caractéristiques à extraire du DataFrame.
        root_dir (str, optional) : Le répertoire racine où les fichiers CSV sont stockés. Par défaut, "data".

    Retour :
        tuple : Un tuple contenant deux listes :
                - liste_X (list) : Liste des caractéristiques extraites des fichiers CSV.
                - liste_Y (list) : Liste des labels de clusters extraits des fichiers CSV.
    """
    
    data_path = Path(root_dir)
    file_paths = list(data_path.glob('**/*.csv'))
    
    for file in file_paths:  

        df = pd.read_csv(file, header=0)

        X = df.iloc[:,:features]
        y = df['Cluster_Label']
        

        liste_X=X.values.tolist()
        liste_Y=y.tolist()
    

    return liste_X,liste_Y

def show_plot(x,y,n_components):
    """
    Affiche un graphique 2D des données, colorié par cluster, en utilisant "matplotlib".

    Cette fonction prend les données d'entrée (caractéristiques et labels de clusters) et génère un nuage de points
    en 2D, où chaque cluster est affiché dans une couleur différente. Les points sont répartis selon leurs 
    caractéristiques et colorés en fonction de leur label de cluster.

    Arguments :
        x (list or np.array) : Liste ou tableau des caractéristiques des échantillons (dimensions 2D).
        y (list or np.array) : Liste ou tableau des labels de clusters associés aux échantillons.
        n_components (int) : Le nombre de clusters (composants) à afficher, qui détermine le nombre de couleurs 
                             utilisées pour l'affichage.

    Retour :
        None
    """
    plt.figure(1)
    colors = []

    x = np.array(x)
    y = np.array(y)
    for i in range(n_components):
        hex = '#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) 
        colors.append(hex)

    for k, col in enumerate(colors):
        cluster_data = y == k # Varie en fonction de n_components
        plt.scatter(x[cluster_data, 0], x[cluster_data, 1], c=col, marker=".", s=10)

    # plt.scatter(x,y,c="b", s=50)
    plt.title("Jeu test")
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == "__main__":

    N_SAMPLES = 50
    cluster = 2
    features = 5

    dir=create_directory(n_components=cluster)
    df=create_data(N_SAMPLES,features,cluster)
    save_data(features,df,dir)

    x,y = load_data(features, "test")

    show_plot(x,y,cluster)
