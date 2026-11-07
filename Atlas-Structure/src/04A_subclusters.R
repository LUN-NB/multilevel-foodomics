# 04A fixed PAM subcluster assignments from frozen score inputs
# Core path entry: resolve project root from Rscript, independent of caller cwd.
.atlas_file <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(.atlas_file) != 1L) stop("Run with Rscript.")
.atlas_dir <- dirname(normalizePath(sub("^--file=", "", .atlas_file), mustWork = TRUE))
setwd(normalizePath(file.path(.atlas_dir, ".."), mustWork = TRUE))
source(file.path("configs", "paths.R"))


output_dir <- file.path(run_dir, "04A")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

CFG <- list(
  seed = 20260802,
  gsva_heatmap_top_n = 50,
  gsva_compound_k_by_macro = c("MacroGroup 1" = 2L, "MacroGroup 2" = 2L)
)

pathway_row_z <- function(score_mat) {
  row_mean <- rowMeans(score_mat)
  row_sd <- apply(score_mat, 1, stats::sd)
  valid <- is.finite(row_mean) & is.finite(row_sd) & row_sd > 0
  score_mat <- score_mat[valid, , drop = FALSE]
  z_mat <- sweep(score_mat, 1, row_mean[valid], FUN = "-")
  z_mat <- sweep(z_mat, 1, row_sd[valid], FUN = "/")
  z_mat
}

# Use 1 minus Spearman correlation to compare compound profiles over the selected features.
make_spearman_distance <- function(x) {
  x <- as.matrix(x)
  if (nrow(x) < 2L || ncol(x) < 2L) {
    stop("Spearman distance requires at least two items and two features.")
  }
  rho <- suppressWarnings(stats::cor(
    t(x),
    method = "spearman",
    use = "pairwise.complete.obs"
  ))
  rho[!is.finite(rho)] <- 0
  rho[rho > 1] <- 1
  rho[rho < -1] <- -1
  diag(rho) <- 1
  stats::as.dist(1 - rho)
}

# Use fixed-K PAM with complete linkage as the fallback.
partition_compounds <- function(x, k) {
  d <- make_spearman_distance(x)
  pam_result <- tryCatch(
    cluster::pam(d, k = k, diss = TRUE, cluster.only = TRUE),
    error = function(e) NULL
  )
  if (!is.null(pam_result)) {
    return(as.integer(pam_result))
  }
  as.integer(stats::cutree(
    stats::hclust(d, method = "complete"),
    k = k
  ))
}

# Assign cluster IDs in Ward tree order of the centroids.
relabel_clusters_by_centroid <- function(labels, x) {
  labels <- as.integer(labels)
  x <- as.matrix(x)
  old_levels <- sort(unique(labels))
  centers <- do.call(rbind, lapply(old_levels, function(k) {
    colMeans(x[labels == k, , drop = FALSE], na.rm = TRUE)
  }))
  center_order <- if (length(old_levels) > 1L) {
    stats::hclust(
      stats::dist(centers),
      method = "ward.D2"
    )$order
  } else {
    seq_along(old_levels)
  }
  ordered_old_levels <- old_levels[center_order]
  old_to_new <- setNames(seq_along(ordered_old_levels), ordered_old_levels)
  as.integer(old_to_new[as.character(labels)])
}

get_fixed_compound_k <- function(group_name, n_compounds) {
  fixed_k <- unname(CFG$gsva_compound_k_by_macro[group_name])
  if (length(fixed_k) != 1L || !is.finite(fixed_k)) {
    stop("No fixed compound K configured for ", group_name)
  }
  fixed_k <- as.integer(fixed_k)
  if (fixed_k < 2L || fixed_k >= n_compounds) {
    stop(
      "Invalid fixed compound K for ", group_name,
      ": K = ", fixed_k, "; compounds = ", n_compounds
    )
  }
  fixed_k
}

cluster_group <- function(group_name, score_mat) {
  cluster_z_mat <- pathway_row_z(score_mat)
  if (nrow(cluster_z_mat) == 0L) {
    stop(group_name, ": no variable score features are available.")
  }

  pathway_mad <- apply(
    score_mat[rownames(cluster_z_mat), , drop = FALSE],
    1,
    stats::mad
  )
  heatmap_pathways <- names(sort(pathway_mad, decreasing = TRUE))
  heatmap_pathways <- head(
    heatmap_pathways,
    min(CFG$gsva_heatmap_top_n, length(heatmap_pathways))
  )
  cluster_z_mat <- cluster_z_mat[heatmap_pathways, , drop = FALSE]
  compound_space <- t(cluster_z_mat)
  fixed_k <- get_fixed_compound_k(group_name, nrow(compound_space))
  set.seed(CFG$seed)
  fixed_labels <- partition_compounds(compound_space, k = fixed_k)
  fixed_labels <- relabel_clusters_by_centroid(fixed_labels, compound_space)
  names(fixed_labels) <- rownames(compound_space)
  compound_cluster_levels <- paste0("Compound cluster ", seq_len(fixed_k))
  compound_cluster <- factor(
    paste0("Compound cluster ", fixed_labels),
    levels = compound_cluster_levels
  )
  names(compound_cluster) <- names(fixed_labels)
  # Order clusters by their centroids and order compounds within each cluster by Spearman distance.
  compound_order <- unlist(lapply(compound_cluster_levels, function(level_name) {
    cids <- names(compound_cluster)[compound_cluster == level_name]
    if (length(cids) <= 2L) {
      return(cids)
    }
    hc <- stats::hclust(
      make_spearman_distance(t(cluster_z_mat[, cids, drop = FALSE])),
      method = "complete"
    )
    cids[hc$order]
  }), use.names = FALSE)
  compound_cluster <- compound_cluster[compound_order]
  compound_cluster_df <- data.frame(
    CID = compound_order,
    FixedPAMCluster = as.character(compound_cluster),
    CompoundCluster = as.character(compound_cluster),
    Fixed_K = fixed_k,
    Distance = "1 - Spearman correlation",
    Algorithm = "PAM; complete-linkage fallback",
    stringsAsFactors = FALSE
  )
  compound_cluster_df
}

for (group in 1:2) {
  group_tag <- paste0("MacroGroup_", group)
  score_mat <- as.matrix(read.csv(
    file.path(score_dir, paste0(group_tag, "_RawScores.csv")),
    row.names = 1,
    check.names = FALSE
  ))
  clusters <- cluster_group(paste("MacroGroup", group), score_mat)
  write.csv(
    clusters,
    file.path(output_dir, paste0(group_tag, "_CompoundClusters.csv")),
    row.names = FALSE
  )
}
