import pandas as pd
import numpy as np
import os
import re
import glob
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
except ImportError:
    print("ATTENTION : La librairie 'statsmodels' n'a pas été trouvée. Les étapes d'ANOVA ne pourront pas s'exécuter.")
    ols = anova_lm = None


OUTPUT_DIR = "results_mean"

print("ÉTAPE 1 : Lecture et Consolidation des Fichiers de Moyennes")

def load_and_consolidate_data():
    """
    Charge tous les fichiers CSV du répertoire OUTPUT_DIR, les combine,
    les pivote et prépare un DataFrame pour l'ANOVA.

    Cette fonction recherche tous les fichiers CSV dans le répertoire
    spécifié, les combine en un DataFrame, et les transforme en format long
    pour une analyse ultérieure via ANOVA.

    Arguments :
        Aucun

    Retour :
        DataFrame : Un DataFrame consolidé et pivoté, prêt pour l'ANOVA.
    """
    
    all_mean_files = glob.glob(os.path.join(OUTPUT_DIR, '**', '*.csv'), recursive=True)

    if not all_mean_files:
        raise Exception("\n[ERREUR FATALE] Aucun fichier CSV trouvé dans le dossier 'results_mean'.")

    combined_data_wide_list = []

    for file_path in all_mean_files:
        try:
            data = pd.read_csv(
                file_path, 
                decimal='.', 
                dtype={'Cluster': 'category', 'Features': 'category'}
            )
            
            match = re.search(r'results_mean[/\\](.*?)[/\\]', file_path, re.IGNORECASE)
            metric_type = match.group(1) if match else "Autre"
            metric_type = 'tc' if metric_type.lower() == 'trustworthiness' else metric_type.lower()
            
            data['Métrique'] = metric_type
            combined_data_wide_list.append(data)
            
        except Exception as e:
            print(f"  [ALERTE] Impossible de lire ou traiter le fichier {file_path}: {e}")
            continue

    if not combined_data_wide_list:
        raise Exception("\n[ERREUR FATALE] Les fichiers de moyennes sont vides ou illisibles.")
        
    combined_data_wide = pd.concat(combined_data_wide_list, ignore_index=True)
    
    id_cols = ['Cluster', 'Features', 'Métrique']
    score_cols = [col for col in combined_data_wide.columns if col not in id_cols and col not in ['K_Neighbors']]

    print(f"Fichiers de moyennes chargés. Dimensions initiales: {len(combined_data_wide)} lignes, {len(combined_data_wide.columns)} colonnes.")

    final_data_long = combined_data_wide.melt(
        id_vars=id_cols,
        value_vars=score_cols,
        var_name='Nom_Colonne_Originale',
        value_name='Score'
    )
    
    if final_data_long.empty:
        raise Exception("\n[ERREUR FATALE] Le pivotement n'a produit aucune ligne de données.")
    
    final_data_long['Parametre_TSNE'] = final_data_long['Nom_Colonne_Originale'].str.extract(r'(TSNE_P\d+|Original|P\d+)', expand=False).fillna('Original')

    final_data_long['Métrique_Analyse'] = np.select(
        [
            final_data_long['Nom_Colonne_Originale'].str.contains('NH', na=False),
            final_data_long['Nom_Colonne_Originale'].str.contains('Spearman', na=False),
            final_data_long['Nom_Colonne_Originale'].str.contains('Trustworthiness', na=False),
            final_data_long['Nom_Colonne_Originale'].str.contains('Continuity', na=False)
        ],
        ['NH', 'Spearman', 'Trustworthiness', 'Continuity'],
        default='Autre'
    )
    
    data_for_anova = final_data_long.copy()
    data_for_anova = data_for_anova[data_for_anova['Métrique_Analyse'] != 'Autre']
    data_for_anova = data_for_anova.dropna(subset=['Score'])
    
    def simplify_param(p):
        if pd.isna(p): return p
        if p == 'Original': return 'Original'
        return re.sub(r'TSNE_', '', p)
    data_for_anova['Parametre_TSNE_Simplifie'] = data_for_anova['Parametre_TSNE'].apply(simplify_param)
    
    print(f"Pivotement réussi. Lignes utilisées pour l'ANOVA (non agrégées): {len(data_for_anova)} lignes.")

    return data_for_anova

try:
    data_for_anova = load_and_consolidate_data() 

    if ols and anova_lm:
        print("ÉTAPE 2 : ANOVA sur le Score (Métriques vs Perplexité t-SNE)")

        data_for_anova['Parametre_TSNE'] = data_for_anova['Parametre_TSNE'].astype('category')
        data_for_anova['Métrique_Analyse'] = data_for_anova['Métrique_Analyse'].astype('category')

        formula = 'Score ~ C(Métrique_Analyse) * C(Parametre_TSNE)'
        lm = ols(formula, data=data_for_anova).fit()
        
        print("\n--- RÉSULTATS ANOVA (Métrique et Paramètre de Perplexité) ---")
        anova_table = anova_lm(lm)
        print(anova_table)
    else:
        print("\n[ALERTE] L'ANOVA n'a pas été exécutée car 'statsmodels' est manquant.")

    print("ÉTAPE 3 : Création des Graphiques/visualisatio ")
    
    plot_data = data_for_anova.groupby(['Métrique_Analyse', 'Parametre_TSNE_Simplifie'])['Score'].mean().reset_index()
    plot_data = plot_data.rename(columns={'Métrique_Analyse': 'Métrique', 'Score': 'Moyenne_Globale'})

    order_items = plot_data['Parametre_TSNE_Simplifie'].unique()
    def sort_key(x):
        if x == 'Original': return (0, 0)
        num_match = re.search(r'\d+', x)
        num = int(num_match.group(0)) if num_match else 999 
        return (1, num) 

    order = sorted(order_items, key=sort_key)

    plot_data['Parametre_TSNE_Simplifie'] = pd.Categorical(
        plot_data['Parametre_TSNE_Simplifie'], 
        categories=order, 
        ordered=True
    )
    plot_data = plot_data.sort_values('Parametre_TSNE_Simplifie')


    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=plot_data, 
        x='Parametre_TSNE_Simplifie', 
        y='Moyenne_Globale', 
        hue='Métrique', 
        marker='o', 
        dashes=False,
        errorbar=None, 
        palette='viridis',
    )
    plt.title("Line Plot : Impact de la Perplexité t-SNE sur la Qualité (Moyenne Globale)", fontsize=14)
    plt.xlabel("Paramètre de Perplexité t-SNE", fontsize=12)
    plt.ylabel("Valeur Moyenne Globale de la Métrique", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Métrique de Qualité")
    plt.tight_layout()
    plt.show()


    print("\n-> Génération du Heatmap...")
    heatmap_data = plot_data.pivot(
        index='Métrique', 
        columns='Parametre_TSNE_Simplifie', 
        values='Moyenne_Globale'
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=".3f", 
        cmap="YlGnBu", 
        linewidths=.5, 
        cbar_kws={'label': 'Moyenne Globale du Score'}
    )
    plt.title("Heatmap : Scores Moyens (Métrique vs. Perplexité)", fontsize=14)
    plt.ylabel("Métrique de Qualité")
    plt.xlabel("Paramètre de Perplexité t-SNE")
    plt.tight_layout()
    plt.show()


    print("\n-> Génération du Box Plot...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=data_for_anova,
        x='Parametre_TSNE_Simplifie', 
        y='Score', 
        palette='Set2',
        order=order
    )
    plt.title("Box Plot : Distribution des Scores de Qualité par Paramètre de Perplexité", fontsize=14)
    plt.xlabel("Paramètre de Perplexité t-SNE", fontsize=12)
    plt.ylabel("Score de Qualité (Observations)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"\n[FATAL] Une erreur s'est produite durant l'exécution: {e}")