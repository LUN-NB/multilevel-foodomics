# 03B structural diagnosis and MacroGroup partition
# Core path entry: resolve project root from Rscript, independent of caller cwd.
.atlas_file <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(.atlas_file) != 1L) stop("Run with Rscript.")
.atlas_dir <- dirname(normalizePath(sub("^--file=", "", .atlas_file), mustWork = TRUE))
setwd(normalizePath(file.path(.atlas_dir, ".."), mustWork = TRUE))
source(file.path("configs", "paths.R"))


set.seed(20260802)

library(cluster)
library(clValid)
library(igraph)
library(irlba)
library(mclust)
library(Seurat)

matrix_path <- rowz_csv
output_dir <- file.path(run_dir, "03B")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

seed <- 20260802
n_pc_global <- 30
n_pc_within <- 30
k_grid <- 2:8
global_snn_k <- 20
resolution_grid <- c(
  0.01,
  seq(0.05, 0.50, by = 0.05),
  seq(0.60, 1.00, by = 0.10)
)
umap_min_dist <- 0.30
diffusion_k <- 10

final_df <- read.csv(
  matrix_path,
  row.names = 1,
  check.names = FALSE,
  stringsAsFactors = FALSE
)
final_mat <- as.matrix(final_df)
storage.mode(final_mat) <- "double"
final_mat <- final_mat[, apply(final_mat, 2, var) > 0, drop = FALSE]

mean_silhouette <- function(labels, distance) {
  mean(cluster::silhouette(as.integer(factor(labels)), distance)[, "sil_width"])
}

relabel_by_size <- function(labels) {
  label_order <- names(sort(table(labels), decreasing = TRUE))
  match(as.character(labels), label_order)
}

cluster_sizes <- function(labels) {
  paste(sort(as.integer(table(labels)), decreasing = TRUE), collapse = "/")
}

pairwise_ari_mean <- function(label_list) {
  pairs <- combn(seq_along(label_list), 2)
  mean(apply(pairs, 2, function(x) {
    adjustedRandIndex(label_list[[x[1]]], label_list[[x[2]]])
  }))
}

knn_adjacency <- function(distance_matrix, k, mutual = FALSE) {
  work <- distance_matrix
  diag(work) <- Inf
  neighbors <- t(apply(work, 1, function(x) order(x)[seq_len(k)]))
  adjacency <- matrix(FALSE, nrow(work), nrow(work))
  for (i in seq_len(nrow(work))) adjacency[i, neighbors[i, ]] <- TRUE
  if (mutual) adjacency & t(adjacency) else adjacency | t(adjacency)
}

largest_component_fraction <- function(adjacency) {
  graph <- igraph::graph_from_adjacency_matrix(
    adjacency,
    mode = "undirected",
    diag = FALSE
  )
  max(igraph::components(graph)$csize) / igraph::vcount(graph)
}

local_diffusion_map <- function(pc, k, dimensions = 5) {
  distance_matrix <- as.matrix(dist(pc))
  work <- distance_matrix
  diag(work) <- Inf
  local_scale <- apply(work, 1, function(x) sort(x)[k])
  adjacency <- knn_adjacency(distance_matrix, k, mutual = FALSE)
  affinity <- exp(-(distance_matrix^2) / outer(local_scale, local_scale)) * adjacency
  diag(affinity) <- 0
  density <- rowSums(affinity)
  kernel <- affinity / outer(density, density)
  degree_value <- rowSums(kernel)
  symmetric_operator <- kernel / sqrt(outer(degree_value, degree_value))
  eig <- eigen(symmetric_operator, symmetric = TRUE)
  keep <- 2:(dimensions + 1)
  eigenvalues <- eig$values[keep]
  eigenvectors <- eig$vectors[, keep, drop = FALSE] / sqrt(degree_value)
  coordinates <- sweep(eigenvectors, 2, eigenvalues, "*")
  colnames(coordinates) <- paste0("DC", seq_len(dimensions))
  if (cor(coordinates[, 1], pc[, 1]) < 0) coordinates[, 1] <- -coordinates[, 1]
  list(
    coordinates = coordinates,
    eigenvalues = eigenvalues,
    distance_matrix = distance_matrix
  )
}

mst_diagnostics <- function(distance_matrix) {
  graph <- igraph::graph_from_adjacency_matrix(
    distance_matrix,
    mode = "undirected",
    weighted = TRUE,
    diag = FALSE
  )
  tree <- igraph::mst(graph, weights = igraph::E(graph)$weight)
  edge_weights <- igraph::E(tree)$weight
  cut_tree <- igraph::delete_edges(
    tree,
    igraph::E(tree)[which.max(edge_weights)]
  )
  cut_sizes <- sort(igraph::components(cut_tree)$csize, decreasing = TRUE)
  list(
    max_to_median = max(edge_weights) / median(edge_weights),
    smaller_side_fraction = min(cut_sizes) / igraph::vcount(tree),
    edges = matrix(
      as.integer(igraph::ends(tree, igraph::E(tree), names = FALSE)),
      ncol = 2
    )
  )
}

format_number <- function(x, digits = 3) {
  ifelse(is.na(x), "", formatC(x, digits = digits, format = "f"))
}

centered_mat <- scale(final_mat, center = TRUE, scale = FALSE)
profile_assay <- SeuratObject::CreateAssay5Object(data = t(final_mat))
profile_assay$scale.data <- t(centered_mat)
seurat_object <- SeuratObject::CreateSeuratObject(
  counts = profile_assay,
  assay = "RNA",
  project = "CPI-MEC_Atlas"
)
VariableFeatures(seurat_object) <- rownames(seurat_object)

seurat_object <- Seurat::RunPCA(
  seurat_object,
  assay = "RNA",
  features = rownames(seurat_object),
  npcs = n_pc_global,
  seed.use = seed,
  verbose = FALSE
)
global_pc <- SeuratObject::Embeddings(
  seurat_object,
  reduction = "pca"
)[, seq_len(n_pc_global), drop = FALSE]
global_distance <- dist(global_pc)
global_tree <- hclust(global_distance, method = "ward.D2")

set.seed(seed)
clvalid_result <- withCallingHandlers(
  clValid(
    global_pc,
    nClust = k_grid,
    clMethods = c("hierarchical", "kmeans", "pam"),
    validation = c("internal", "stability"),
    metric = "euclidean",
    method = "ward"
  ),
  warning = function(w) {
    if (grepl("ward.*renamed", conditionMessage(w), ignore.case = TRUE)) {
      invokeRestart("muffleWarning")
    }
  }
)

clvalid_optimal <- as.data.frame(optimalScores(clvalid_result))
clvalid_optimal$Validation_index <- rownames(clvalid_optimal)
rownames(clvalid_optimal) <- NULL
clvalid_optimal$Clusters <- as.integer(as.character(clvalid_optimal$Clusters))

clvalid_vote_table <- as.data.frame(table(
  factor(clvalid_optimal$Clusters, levels = k_grid)
))
names(clvalid_vote_table) <- c("K", "clValid_votes")
clvalid_vote_table$K <- as.integer(as.character(clvalid_vote_table$K))
selected_global_k <- min(clvalid_vote_table$K[
  clvalid_vote_table$clValid_votes == max(clvalid_vote_table$clValid_votes)
])
clvalid_vote_table$Selected <- clvalid_vote_table$K == selected_global_k

clvalid_array <- clValid::measures(clvalid_result)
clvalid_long <- expand.grid(
  Validation_index = dimnames(clvalid_array)[[1]],
  K = as.integer(dimnames(clvalid_array)[[2]]),
  Method = dimnames(clvalid_array)[[3]],
  stringsAsFactors = FALSE
)
clvalid_long$Value <- as.vector(clvalid_array)
clvalid_long$Validation_type <- ifelse(
  clvalid_long$Validation_index %in% c("Connectivity", "Dunn", "Silhouette"),
  "Internal",
  "Stability"
)

global_scan <- do.call(rbind, lapply(k_grid, function(k) {
  set.seed(seed + k)
  labels <- list(
    Kmeans = kmeans(global_pc, centers = k, nstart = 100)$cluster,
    PAM = pam(global_distance, k = k, diss = TRUE)$clustering,
    Ward = cutree(global_tree, k = k)
  )
  data.frame(
    k = k,
    Kmeans = mean_silhouette(labels$Kmeans, global_distance),
    PAM = mean_silhouette(labels$PAM, global_distance),
    Ward = mean_silhouette(labels$Ward, global_distance),
    Cross_method_ARI = pairwise_ari_mean(labels)
  )
}))

set.seed(seed)
reference_labels <- list(
  Kmeans = kmeans(global_pc, centers = selected_global_k, nstart = 200)$cluster,
  PAM = pam(global_distance, k = selected_global_k, diss = TRUE)$clustering,
  Ward = cutree(global_tree, k = selected_global_k)
)
global_cross_method_ari <- pairwise_ari_mean(reference_labels)

# ------------------------------------------------------------
# 3. Seurat graph clustering and UMAP
# ------------------------------------------------------------

seurat_object <- Seurat::FindNeighbors(
  seurat_object,
  reduction = "pca",
  dims = seq_len(n_pc_global),
  k.param = global_snn_k,
  graph.name = c("RNA_nn", "RNA_snn"),
  verbose = FALSE
)
seurat_object <- Seurat::FindClusters(
  seurat_object,
  graph.name = "RNA_snn",
  algorithm = 4,
  leiden_method = "igraph",
  resolution = resolution_grid,
  random.seed = seed,
  verbose = FALSE
)
seurat_object <- Seurat::RunUMAP(
  seurat_object,
  reduction = "pca",
  dims = seq_len(n_pc_global),
  n.neighbors = global_snn_k,
  metric = "cosine",
  min.dist = umap_min_dist,
  seed.use = seed,
  verbose = FALSE
)

resolution_columns <- grep(
  "^RNA_snn_res\\.",
  colnames(seurat_object@meta.data),
  value = TRUE
)
leiden_scan <- do.call(rbind, lapply(resolution_columns, function(column) {
  labels <- seurat_object@meta.data[[column]]
  n_clusters <- length(unique(labels))
  data.frame(
    resolution = as.numeric(sub("^RNA_snn_res\\.", "", column)),
    metadata_column = column,
    n_clusters = n_clusters,
    silhouette = if (n_clusters > 1) {
      mean_silhouette(labels, global_distance)
    } else {
      NA_real_
    },
    cluster_sizes = cluster_sizes(labels),
    stringsAsFactors = FALSE
  )
}))
leiden_scan <- leiden_scan[order(leiden_scan$resolution), ]

candidate_rows <- which(leiden_scan$n_clusters == selected_global_k)
selected_scan_row <- candidate_rows[
  which.max(leiden_scan$silhouette[candidate_rows])
]
selected_resolution <- leiden_scan$resolution[selected_scan_row]
selected_resolution_column <- leiden_scan$metadata_column[selected_scan_row]
global_leiden <- seurat_object@meta.data[[selected_resolution_column]]
macrogroup <- relabel_by_size(global_leiden)
names(macrogroup) <- colnames(seurat_object)
seurat_object$MacroGroup <- factor(macrogroup)
Idents(seurat_object) <- "MacroGroup"

global_leiden_reference_ari <- mean(vapply(
  reference_labels,
  function(x) adjustedRandIndex(macrogroup, x),
  numeric(1)
))

global_umap <- SeuratObject::Embeddings(seurat_object, reduction = "umap")
colnames(global_umap) <- c("UMAP1", "UMAP2")

global_partition <- data.frame(
  CID = rownames(final_mat),
  MacroGroup = macrogroup,
  UMAP1 = global_umap[, 1],
  UMAP2 = global_umap[, 2],
  stringsAsFactors = FALSE
)
write.csv(
  global_partition,
  file.path(output_dir, "global_Leiden_MacroGroup_partition.csv"),
  row.names = FALSE
)

# ------------------------------------------------------------
# 4. Structure within each MacroGroup
# ------------------------------------------------------------

analyze_macrogroup <- function(group_id) {
  cid <- names(macrogroup)[macrogroup == group_id]
  group_mat <- final_mat[cid, , drop = FALSE]
  group_mat <- group_mat[, apply(group_mat, 2, var) > 0, drop = FALSE]
  n_pc <- min(n_pc_within, nrow(group_mat) - 1)
  set.seed(seed + group_id)
  pca_fit <- prcomp_irlba(group_mat, n = n_pc, center = TRUE, scale. = FALSE)
  pc <- pca_fit$x[, seq_len(n_pc), drop = FALSE]
  pc_distance <- dist(pc)
  pc_distance_matrix <- as.matrix(pc_distance)
  diffusion <- local_diffusion_map(pc, diffusion_k, dimensions = 5)
  dc <- diffusion$coordinates
  rownames(dc) <- cid
  profile_distance_matrix <- 1 - cor(t(group_mat), method = "pearson")
  profile_distance <- as.dist(profile_distance_matrix)
  dc_order <- order(dc[, 1])
  consecutive_profile_distance <- vapply(
    seq_len(length(dc_order) - 1),
    function(i) profile_distance_matrix[dc_order[i], dc_order[i + 1]],
    numeric(1)
  )
  mst_result <- mst_diagnostics(pc_distance_matrix)
  subcluster_scan <- do.call(rbind, lapply(k_grid, function(k) {
    set.seed(seed + group_id * 100 + k)
    labels <- list(
      Kmeans = kmeans(pc, centers = k, nstart = 100)$cluster,
      PAM = pam(pc_distance, k = k, diss = TRUE)$clustering,
      Ward = cutree(hclust(pc_distance, method = "ward.D2"), k = k)
    )
    silhouettes <- vapply(labels, mean_silhouette, numeric(1), distance = pc_distance)
    data.frame(
      k = k,
      mean_silhouette = mean(silhouettes),
      cross_method_ARI = pairwise_ari_mean(labels)
    )
  }))
  best_row <- subcluster_scan[which.max(subcluster_scan$mean_silhouette), ]
  list(
    group_id = group_id,
    n = nrow(group_mat),
    dc = dc,
    diffusion_eigenvalues = diffusion$eigenvalues,
    mutual_lcc = largest_component_fraction(
      knn_adjacency(pc_distance_matrix, diffusion_k, mutual = TRUE)
    ),
    consecutive_ratio = mean(consecutive_profile_distance) / median(profile_distance),
    mst = mst_result,
    subcluster_scan = subcluster_scan,
    best_k = best_row$k,
    best_silhouette = best_row$mean_silhouette,
    best_cross_method_ari = best_row$cross_method_ARI
  )
}

group_ids <- sort(unique(macrogroup))
within_results <- lapply(group_ids, analyze_macrogroup)
names(within_results) <- paste0("MacroGroup_", group_ids)

# ------------------------------------------------------------
# 5. Core evidence tables
# ------------------------------------------------------------

write.csv(
  clvalid_vote_table,
  file.path(output_dir, "clValid_K_votes.csv"),
  row.names = FALSE
)
write.csv(
  clvalid_optimal,
  file.path(output_dir, "clValid_optimal_scores.csv"),
  row.names = FALSE
)

global_evidence <- data.frame(
  Metric = c(
    "clValid-selected K",
    "Leiden partition sizes",
    "Selected Leiden resolution",
    "Cross-method ARI at selected K",
    "Leiden agreement with reference algorithms"
  ),
  Value = c(
    selected_global_k,
    cluster_sizes(macrogroup),
    format_number(selected_resolution, 2),
    format_number(global_cross_method_ari),
    format_number(global_leiden_reference_ari)
  ),
  stringsAsFactors = FALSE
)

within_evidence <- do.call(rbind, lapply(within_results, function(x) {
  data.frame(
    MacroGroup = x$group_id,
    N = x$n,
    Mutual_kNN_LCC = x$mutual_lcc,
    Best_internal_K = x$best_k,
    Best_silhouette = x$best_silhouette,
    Cross_method_ARI = x$best_cross_method_ari,
    DC1_neighbor_distance_ratio = x$consecutive_ratio,
    MST_max_median_edge_ratio = x$mst$max_to_median,
    stringsAsFactors = FALSE
  )
}))

write.csv(
  global_evidence,
  file.path(output_dir, "global_structure_core_evidence.csv"),
  row.names = FALSE
)
write.csv(
  within_evidence,
  file.path(output_dir, "within_group_structure_core_evidence.csv"),
  row.names = FALSE
)

write.csv(clvalid_long, file.path(output_dir, "clValid_scan.csv"), row.names = FALSE)
write.csv(global_scan, file.path(output_dir, "global_partition_scan.csv"), row.names = FALSE)
write.csv(leiden_scan, file.path(output_dir, "leiden_resolution_scan.csv"), row.names = FALSE)
