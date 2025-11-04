# Chargement du fichier CSV contenant les résultats
data <- read.csv("merged_all_results.csv")

# Affichage des noms des colonnes
names(data)

# Calcul de la moyenne des métriques par centre, nombre de features, perplexité et répétition
library(dplyr)
data_mean_rep <- data %>%
  group_by(center, features, perplexity, repetition) %>%
  summarise(
    mean_trust = mean(trustworthiness, na.rm = TRUE),
    mean_cont  = mean(continuity,      na.rm = TRUE),
    mean_spear = mean(spearman,        na.rm = TRUE),
    mean_knn   = mean(nh_score,        na.rm = TRUE),
    .groups = "drop"
  )

# Test de Bartlett pour vérifier l’homogénéité des variances
bartlett.test(mean_trust ~ interaction(features, perplexity), data = data_mean_rep)

# Tests de Welch (ANOVA) pour comparer les moyennes selon les conditions
welch_trust <- oneway.test(mean_trust ~ interaction(features, perplexity),
                           data = data_mean_rep, var.equal = FALSE)
print(welch_trust)

welch_cont <- oneway.test(mean_cont ~ interaction(features, perplexity),
                          data = data_mean_rep, var.equal = FALSE)
print(welch_cont)

welch_spear <- oneway.test(mean_spear ~ interaction(features, perplexity),
                           data = data_mean_rep, var.equal = FALSE)
print(welch_spear)

welch_knn <- oneway.test(mean_knn ~ interaction(features, perplexity),
                         data = data_mean_rep, var.equal = FALSE)
print(welch_knn)

# Chargement des bibliothèques pour la visualisation
library(ggplot2)
library(viridis)

# Calcul des moyennes et écarts-types par configuration (features, perplexity)
summary_stats <- data_mean_rep %>%
  group_by(features, perplexity) %>%
  summarise(
    mean_trust = mean(mean_trust, na.rm = TRUE),
    sd_trust   = sd(mean_trust,   na.rm = TRUE),
    mean_cont  = mean(mean_cont,  na.rm = TRUE),
    sd_cont    = sd(mean_cont,    na.rm = TRUE),
    mean_spear = mean(mean_spear, na.rm = TRUE),
    sd_spear   = sd(mean_spear,   na.rm = TRUE),
    mean_knn   = mean(mean_knn,   na.rm = TRUE),
    sd_knn     = sd(mean_knn,     na.rm = TRUE),
    .groups = "drop"
  )

# Fonction pour tracer une métrique avec moyenne et écart-type
plot_metric <- function(df, metric_mean, metric_sd, title, y_label) {
  ggplot(df, aes(
    x = as.numeric(as.character(features)),
    y = !!sym(metric_mean),
    color = as.factor(perplexity),
    group = as.factor(perplexity)
  )) +
    geom_line(linewidth = 1) +
    geom_point(size = 2) +
    geom_errorbar(
      aes(ymin = !!sym(metric_mean) - !!sym(metric_sd),
          ymax = !!sym(metric_mean) + !!sym(metric_sd)),
      width = 1, alpha = 0.5
    ) +
    scale_color_viridis_d(option = "C", end = 0.9) +
    theme_minimal(base_size = 14) +
    labs(
      title = title,
      x = "Nombre de dimensions (features)",
      y = y_label,
      color = "Perplexité"
    ) +
    theme(
      legend.position = "right",
      plot.title = element_text(face = "bold", size = 15)
    )
}

# Génération des graphiques pour chaque métrique
plot_trust <- plot_metric(summary_stats, "mean_trust", "sd_trust",
                          "Évolution du Trustworthiness selon la dimension et la perplexité",
                          "Trustworthiness moyen ± écart-type")

plot_cont <- plot_metric(summary_stats, "mean_cont", "sd_cont",
                         "Évolution de la Continuity selon la dimension et la perplexité",
                         "Continuity moyenne ± écart-type")

plot_spear <- plot_metric(summary_stats, "mean_spear", "sd_spear",
                          "Évolution de la corrélation de Spearman selon la dimension et la perplexité",
                          "Corrélation de Spearman moyenne ± écart-type")

plot_knn <- plot_metric(summary_stats, "mean_knn", "sd_knn",
                        "Évolution du score KNN (NH) selon la dimension et la perplexité",
                        "Score KNN moyen ± écart-type")

# Affichage des graphiques
plot_trust
plot_cont
plot_spear
plot_knn
