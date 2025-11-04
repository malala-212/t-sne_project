import data_generation as gen
import t_sne
import os
from pathlib import Path
import pandas as pd
import knn as knn
import spearman as spearman
import trustworthiness_continuity as trust

N_SAMPLES = 5000
NB_CLUSTER_MIN = 1
NB_CLUSTER_MAX = 7
NB_FEATURES_MIN_BOUCLE_1 = 2
NB_FEATURES_MAX_BOUCLE_1 = 50
PAS_BOUCLE_1 = 5
NB_FEATURES_MIN_BOUCLE_2 = 50
NB_FEATURES_MAX_BOUCLE_2 = 101
PAS_BOUCLE_2 = 10
TSNE_N_COMPONENTS = 2
PERPLEXITIES = [5, 15, 20, 50, 90]
K_NEIGHBORS = 20

cluster_range = range(NB_CLUSTER_MIN, NB_CLUSTER_MAX)
features_list = list(range(NB_FEATURES_MIN_BOUCLE_1, NB_FEATURES_MAX_BOUCLE_1, PAS_BOUCLE_1)) + \
                list(range(NB_FEATURES_MIN_BOUCLE_2, NB_FEATURES_MAX_BOUCLE_2, PAS_BOUCLE_2))

def run_pipeline(ROOT_DATA_DIR, ROOT_TSNE_DIR, TC_RESULTS_DIR,
                 NH_RESULTS_DIR, SPEARMAN_RESULTS_DIR, ROOT_IMAGES_DIR):
    """
    Exécute le pipeline de traitement des données : génération, réduction de dimensionnalité (t-SNE),
    et analyses de la fidélité et continuité des résultats, incluant les analyses K-NN et Spearman.

    Cette fonction orchestre les différentes étapes du pipeline, en commençant par la génération des données
    et en passant par plusieurs analyses. Les résultats sont sauvegardés dans les répertoires spécifiés.

    Étapes :
        1. Création des données synthétiques pour chaque combinaison de clusters et de caractéristiques.
        2. Application de l'algorithme t-SNE pour la réduction de dimensionnalité.
        3. Sauvegarde des résultats d'analyse du voisinage (K-NN).
        4. Calcul de la corrélation de Spearman entre les données de t-SNE.
        5. Calcul de la fidélité (trustworthiness) des réductions de dimensionnalité.

    Arguments :
        ROOT_DATA_DIR (str or Path) : Répertoire où les données générées seront stockées.
        ROOT_TSNE_DIR (str or Path) : Répertoire où les résultats de l'algorithme t-SNE seront sauvegardés.
        TC_RESULTS_DIR (str or Path) : Répertoire où les résultats de l'analyse trustworthiness seront sauvegardés.
        NH_RESULTS_DIR (str or Path) : Répertoire où les résultats de l'analyse K-NN seront sauvegardés.
        SPEARMAN_RESULTS_DIR (str or Path) : Répertoire où les résultats de l'analyse de Spearman seront sauvegardés.
        ROOT_IMAGES_DIR (str or Path) : Répertoire où les visualisations des résultats seront générées (images).

    Retour :
        None
    """

    print(" Phase 1: Démarrage de la génération des données ")
    for cluster in cluster_range:
        dir_path = gen.create_directory(cluster, ROOT_DATA_DIR)  
        for feat in features_list:
            df = gen.create_data(N_SAMPLES, feat, cluster)
            gen.save_data(feat, df, dir_path)


    print("\n Phase 2: Démarrage du traitement t-SNE ")
    for cluster in cluster_range:
        for feat in features_list:
            data_dir = Path(ROOT_DATA_DIR) / f'center{cluster}'
            file_path = data_dir / f"features{feat}.csv"
            if file_path.exists():
                df_loaded = pd.read_csv(file_path, header=0)
                X = df_loaded.drop(columns=['Cluster_Label']).values.tolist()
                Y = df_loaded['Cluster_Label'].tolist()
                
                t_sne.test_tsne(
                    x=X,
                    y=Y,
                    n_component=TSNE_N_COMPONENTS,
                    cluster=cluster,
                    features=feat,
                    output_dir=ROOT_TSNE_DIR,
                    image_output_dir=ROOT_IMAGES_DIR,
                )

    print("\nNH ")
    knn.compare_and_save_nh_results(
        cluster_range=cluster_range,
        features_list=features_list,
        perplexities=PERPLEXITIES,
        k_neighbors=K_NEIGHBORS,
        root_data_dir=ROOT_DATA_DIR,
        root_tsne_dir=ROOT_TSNE_DIR,
        nh_results_base_name=NH_RESULTS_DIR,
    )

    print("\n Spearman ")
    spearman.run_spearman_analysis(
        cluster_range=cluster_range,
        features_list=features_list,
        perplexities=PERPLEXITIES,
        root_data_dir=ROOT_DATA_DIR,
        root_tsne_dir=ROOT_TSNE_DIR,
        results_dir=SPEARMAN_RESULTS_DIR
    )

    print("\n Trustworthiness ")
    trust.run_trustworthiness_analysis(
        cluster_range=cluster_range,
        features_list=features_list,
        perplexities=PERPLEXITIES,
        k_neighbors=K_NEIGHBORS,
        root_data_dir=ROOT_DATA_DIR,
        root_tsne_dir=ROOT_TSNE_DIR,
        tc_results_dir=TC_RESULTS_DIR
    )


def main():
    """
    Point d'entrée principal du programme. Cette fonction gère les itérations du pipeline de traitement des données,
    créant des répertoires pour chaque itération et exécutant le pipeline de traitement des données pour chaque ensemble
    de paramètres.

    Pour chaque itération :
        - Crée les répertoires nécessaires pour stocker les résultats.
        - Exécute "run_pipeline" avec les chemins appropriés pour chaque étape du pipeline.

    Arguments :
        None

    Retour :
        None
    """
    for i in range (1,21):

        print(f"\nDÉMARRAGE ITERATION {i} ")
        
        results_dir = Path(f"results{i}")
        (results_dir / "data").mkdir(parents=True, exist_ok=True)
        (results_dir / "data_tsne").mkdir(parents=True, exist_ok=True)
        (results_dir / "trustworthiness").mkdir(parents=True, exist_ok=True)
        (results_dir / "nh").mkdir(parents=True, exist_ok=True)
        (results_dir / "spearman").mkdir(parents=True, exist_ok=True)
        (results_dir / "images").mkdir(parents=True, exist_ok=True)

        run_pipeline(
            ROOT_DATA_DIR=results_dir / "data",
            ROOT_TSNE_DIR=results_dir / "data_tsne",
            TC_RESULTS_DIR=results_dir / "trustworthiness",
            NH_RESULTS_DIR=results_dir / "nh",
            SPEARMAN_RESULTS_DIR=results_dir / "spearman",
            ROOT_IMAGES_DIR=results_dir / "images" 
        )

        print(f" FIN ITERATION {i} (résultats dans {results_dir}) ")

if __name__ == "__main__":
    main()