import numpy as np
from time import time
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from sklearn.manifold import TSNE
import os
import pandas as pd 
import random 
from pathlib import Path 


"""
Code inspiré de https://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html#sphx-glr-auto-examples-manifold-plot-t-sne-perplexity-py
"""


def create_tsne_directory(cluster_nb, features, project_path):
    """
    Crée le répertoire de sortie pour les données t-SNE. 
    La structure du répertoire est : project_path/centerX/featuresY/

    Cette fonction génère un répertoire spécifique pour chaque combinaison de cluster et de caractéristiques.
    Si le répertoire n'existe pas, il est créé.

    Arguments :
        cluster_nb (int) : Le numéro du cluster (ex: 'center1', 'center2', ...).
        features (int) : Le nombre de caractéristiques (features) pour lesquelles les données sont générées.
        project_path (str or Path) : Le répertoire principal où les sous-répertoires de sortie seront créés.

    Retour :
        str : Le chemin complet du répertoire créé pour les données t-SNE des clusters et des caractéristiques spécifiées.
    """

    cluster_dir = Path(project_path) / f'center{cluster_nb}'
    

    output = cluster_dir / f'features{features}'
    
    if not output.exists():
        
        output.mkdir(parents=True, exist_ok=True)
        print(f'Directory "{output}" created')

    return str(output)


def save_tsne_data (cluster, feat, perplexity, df_Y, output):
    """
    Sauvegarde un fichier CSV pour une transformation t-SNE spécifique. Un fichier CSV est créé pour chaque 
    combinaison de caractéristiques et de perplexité.

    Cette fonction enregistre les résultats t-SNE dans un fichier CSV avec un nom unique basé sur le nombre de 
    caractéristiques et la perplexité utilisée. 

    Arguments :
        cluster (int) : Le numéro du cluster pour lequel les données sont générées.
        feat (int) : Le nombre de caractéristiques (features) utilisées pour générer les données.
        perplexity (int) : La perplexité utilisée dans la transformation t-SNE.
        df_Y (pandas.DataFrame) : Le DataFrame contenant les résultats de la transformation t-SNE.
        output (str or Path) : Le répertoire où les résultats t-SNE doivent être sauvegardés.

    Retour :
        None
    """

    try:   
        file_name = f"features{feat}_perplexity{perplexity}.csv" 
        (Path(output) / file_name).write_text(df_Y.to_csv(index=False))

    except Exception as e :
        print("Error saving t-SNE data: ",e)

    else :
        
        print(f"t-SNE data succefully saved for features {feat} and perplexity {perplexity}.")

    return

def test_tsne (x,y,n_component, cluster, features, output_dir, image_output_dir, filename="tsne_result" ):
    """
    Effectue une transformation t-SNE sur les données et génère des graphiques pour visualiser les résultats 
    pour différentes valeurs de perplexité. Un fichier CSV pour chaque perplexité est également sauvegardé.

    Cette fonction applique le t-SNE à un jeu de données avec différentes valeurs de perplexité, génère les 
    projections des données dans un espace de dimension réduite, et crée des visualisations sous forme de graphiques. 
    Les résultats sont sauvegardés dans un répertoire spécifique à la combinaison de cluster et de caractéristiques.

    Arguments :
        x (np.array) : Les caractéristiques des échantillons à projeter en 2D ou plus.
        y (np.array) : Les labels des clusters associés aux échantillons.
        n_component (int) : Le nombre de dimensions (composants) pour la réduction de dimension t-SNE.
        cluster (int) : Le numéro du cluster pour lequel les transformations sont effectuées.
        features (int) : Le nombre de caractéristiques (features) utilisées.
        output_dir (str or Path) : Le répertoire où les résultats t-SNE doivent être sauvegardés.
        image_output_dir (str or Path) : Le répertoire où l'image du graphique doit être sauvegardée.
        filename (str, optional) : Le nom du fichier image de sortie. Par défaut, "tsne_result".

    Retour :
        None
    """

    (fig, subplots) = plt.subplots(1, 6, figsize=(20, 4)) 
    perplexities = [5, 15, 20, 50, 90]
    
    x = np.array(x)
    y_labels = np.array(y) 
    
    unique_clusters = np.unique(y_labels)
    num_unique_clusters = len(unique_clusters)
    
    colors = []
    for _ in range(num_unique_clusters):
        hex_color = '#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) 
        colors.append(hex_color)

    ax = subplots[0]
    for k, col in enumerate(colors):
        cluster_data = y_labels == unique_clusters[k]
        ax.scatter(x[cluster_data, 0], x[cluster_data, 1], c=col, marker=".", s=5, alpha=0.7)
    
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.set_title(f"Original Data\n({features} Features, {cluster} Clusters)") 
    plt.axis("tight")
    
    tsne_output_dir = create_tsne_directory(cluster, features, output_dir) 

    for i, perplexity in enumerate(perplexities):
        ax = subplots[i + 1] 

        t0 = time()
        tsne = TSNE(
            n_components=n_component,
            init="random",
            random_state=0,
            perplexity=perplexity,
        )
        y_tsne = tsne.fit_transform(x) 
        t1 = time()
        print("plot, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
        
        # Création du DataFrame et sauvegarde des données CSV
        df_Y = pd.DataFrame(y_tsne, columns=[f'TSNE_Dim_{j+1}' for j in range(n_component)])
        df_Y['Cluster_Label'] = y_labels
        save_tsne_data(cluster, features, perplexity, df_Y, tsne_output_dir)
        
        ax.set_title("Perplexity=%d" % perplexity)
        for k, col in enumerate(colors):
            cluster_data = y_labels == unique_clusters[k]
            ax.scatter(y_tsne[cluster_data, 0], y_tsne[cluster_data, 1], c=col, marker=".", s=5, alpha=0.7)
        
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis("tight")

    save_current_figure(fig, cluster, features, image_output_dir, filename)
    plt.close()


def save_current_figure(fig,cluster,features,root_folder, filename="tsne_result"):
    """
    Sauvegarde la figure matplotlib dans un sous-dossier spécifique au cluster.

    Cette fonction sauvegarde la figure générée par t-SNE dans un fichier PNG dans le répertoire approprié,
    basé sur le cluster et les caractéristiques, afin de conserver l'organisation des résultats pour chaque cluster.

    Arguments :
        fig (matplotlib.figure.Figure) : La figure à sauvegarder.
        cluster (int) : Le numéro du cluster pour lequel la figure est générée.
        features (int) : Le nombre de caractéristiques utilisées pour générer les données.
        root_folder (str or Path) : Le répertoire de base où la figure sera sauvegardée.
        filename (str, optional) : Le nom de base du fichier de sortie. Par défaut, "tsne_result".

    Retour :
        None
    """
    
    folder = Path(root_folder) / f'center{cluster}'

    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        print(f'Directory "{folder}" created')

    final_filename=f"{filename}_cluster_{cluster}_features_{features}.png"

    filepath = folder / final_filename

    fig.savefig(filepath, dpi=300, bbox_inches="tight") #sauvegarde
    print(f"Figure enregistrée dans : {filepath}")


if __name__ == "__main__":
    from data_generation import load_data

    N_SAMPLES = 50
    cluster = 2
    features = 5
    filename="tsne-test_result"

    x,y = load_data(features, "test")
    test_tsne(x,y,cluster,filename)
