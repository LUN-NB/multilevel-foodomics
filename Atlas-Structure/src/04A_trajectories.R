# 04A within-MacroGroup trajectories
# Core path entry: resolve project root from Rscript, independent of caller cwd.
.atlas_file <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(.atlas_file) != 1L) stop("Run with Rscript.")
.atlas_dir <- dirname(normalizePath(sub("^--file=", "", .atlas_file), mustWork = TRUE))
setwd(normalizePath(file.path(.atlas_dir, ".."), mustWork = TRUE))
source(file.path("configs", "paths.R"))


args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 1L) stop("Supply at most one MacroGroup partition CSV.")
partition_csv <- if (length(args) == 1L) args[[1L]] else frozen_partition
output_dir <- file.path(run_dir, "04A")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

CFG <- list(
  seed = 20260802,
  n_pc = 30,
  umap_n_neighbors = 20,
  umap_min_dist = 0.30,
  dm_n_pcs = 30,
  dm_k = 10,
  mono_cluster_resolution = 0.01,
  mono_cluster_k = 20,
  mono_minimal_branch_len = 5
)
set.seed(CFG$seed)

suppressPackageStartupMessages({
  library(destiny)
  library(dplyr)
  library(monocle3)
  library(Seurat)
  library(SingleCellExperiment)
})

# Some destiny versions still request deprecated Matrix sparse coercions.
# Detect the Matrix.DeprecatedCoerce source rather than a specific target class,
# so dgCMatrix, dgTMatrix, and equivalent variants are covered.
run_diffusion_map_compat <- function(...) {
  withCallingHandlers(
    destiny::DiffusionMap(...),
    warning = function(w) {
      warning_message <- conditionMessage(w)
      warning_call <- paste(
        deparse(conditionCall(w)),
        collapse = ""
      )
      is_deprecated_matrix_coercion <-
        grepl(
          "Matrix.DeprecatedCoerce",
          warning_call,
          fixed = TRUE
        ) ||
        grepl("Matrix-deprecated", warning_message, fixed = TRUE)

      if (is_deprecated_matrix_coercion) {
        tryInvokeRestart("muffleWarning")
      }
    }
  )
}

final_mat <- as.matrix(read.csv(
  rowz_csv,
  row.names = 1,
  check.names = FALSE
))
storage.mode(final_mat) <- "double"
final_mat <- final_mat[, apply(final_mat, 2, var) > 0, drop = FALSE]

global_partition <- read.csv(
  partition_csv,
  stringsAsFactors = FALSE
)
global_partition$CID <- as.character(global_partition$CID)
global_partition$MacroGroup <- factor(
  paste0("MacroGroup ", global_partition$MacroGroup),
  levels = c("MacroGroup 1", "MacroGroup 2")
)

# Fix compound order using the 03B partition
final_mat <- final_mat[global_partition$CID, , drop = FALSE]

run_macrogroup_trajectory <- function(group_name) {
  cat("\n===", group_name, "trajectory ===\n")

  group_cids <- global_partition$CID[
    global_partition$MacroGroup == group_name
  ]
  mat_sub <- final_mat[group_cids, , drop = FALSE]

  # Center protein columns as in 03B
  mat_sub_centered <- scale(mat_sub, center = TRUE, scale = FALSE)

  # Run Seurat PCA using the 03B convention
  profile_assay <- SeuratObject::CreateAssay5Object(data = t(mat_sub))
  profile_assay$scale.data <- t(mat_sub_centered)
  seurat_sub <- SeuratObject::CreateSeuratObject(
    counts = profile_assay,
    assay = "TargetProfile",
    project = "CPI-MEC_Atlas_Trajectory"
  )
  VariableFeatures(seurat_sub) <- rownames(seurat_sub)
  seurat_sub <- Seurat::RunPCA(
    seurat_sub,
    assay = "TargetProfile",
    features = rownames(seurat_sub),
    npcs = CFG$n_pc,
    seed.use = CFG$seed,
    verbose = FALSE
  )

  # Diffusion Map defines trajectory direction
  dm_k_use <- min(CFG$dm_k, floor(nrow(mat_sub) / 3))
  dm_n_pcs_use <- min(CFG$dm_n_pcs, nrow(mat_sub) - 1)
  set.seed(CFG$seed)
  cat("Diffusion Map: running...\n")
  diffusion_map <- run_diffusion_map_compat(
    mat_sub_centered,
    n_pcs = dm_n_pcs_use,
    k = dm_k_use
  )
  cat("Diffusion Map: completed.\n")
  dc_coords <- eigenvectors(diffusion_map)[, 1:3, drop = FALSE]
  rownames(dc_coords) <- rownames(mat_sub)
  colnames(dc_coords) <- c("DC1", "DC2", "DC3")

  # Run Seurat UMAP using the 03B convention
  seurat_sub <- Seurat::RunUMAP(
    seurat_sub,
    reduction = "pca",
    dims = seq_len(CFG$n_pc),
    n.neighbors = CFG$umap_n_neighbors,
    metric = "cosine",
    min.dist = CFG$umap_min_dist,
    seed.use = CFG$seed,
    verbose = FALSE
  )
  umap_sub <- SeuratObject::Embeddings(seurat_sub, reduction = "umap")
  rownames(umap_sub) <- rownames(mat_sub)
  colnames(umap_sub) <- c("UMAP_1", "UMAP_2")

  # Learn the graph on the fixed internal UMAP
  # Shift scores uniformly to nonnegative values for Monocle3 while preserving distances.
  # Shift uniformly to nonnegative values for Monocle3 without changing distances
  expr_sub <- t(mat_sub_centered - min(mat_sub_centered))
  cds <- monocle3::new_cell_data_set(
    expr_sub,
    cell_metadata = data.frame(
      MacroGroup = rep(group_name, ncol(expr_sub)),
      row.names = colnames(expr_sub)
    ),
    gene_metadata = data.frame(
      gene_short_name = rownames(expr_sub),
      row.names = rownames(expr_sub)
    )
  )
  reducedDims(cds)$UMAP <- umap_sub

  mono_k_use <- min(CFG$mono_cluster_k, floor(nrow(mat_sub) / 3))
  set.seed(CFG$seed)
  cds <- monocle3::cluster_cells(
    cds,
    reduction_method = "UMAP",
    resolution = CFG$mono_cluster_resolution,
    k = mono_k_use
  )
  cds <- monocle3::learn_graph(
    cds,
    use_partition = FALSE,
    close_loop = FALSE,
    learn_graph_control = list(
      minimal_branch_len = CFG$mono_minimal_branch_len
    )
  )

  # The minimum-DC1 compound defines the technical root
  root_cid <- rownames(dc_coords)[which.min(dc_coords[, "DC1"])]
  cds <- monocle3::order_cells(cds, root_cells = root_cid)
  pseudotime_values <- monocle3::pseudotime(cds)

  trajectory_df <- as.data.frame(reducedDims(cds)$UMAP)
  colnames(trajectory_df) <- c("UMAP_1", "UMAP_2")
  trajectory_df$CID <- rownames(trajectory_df)
  trajectory_df$MacroGroup <- group_name
  trajectory_df$DC1 <- dc_coords[trajectory_df$CID, "DC1"]
  trajectory_df$DC2 <- dc_coords[trajectory_df$CID, "DC2"]
  trajectory_df$DC3 <- dc_coords[trajectory_df$CID, "DC3"]
  trajectory_df$Pseudotime <- pseudotime_values[trajectory_df$CID]
  trajectory_df$Pseudotime[
    is.infinite(trajectory_df$Pseudotime)
  ] <- NA_real_
  trajectory_df$Is_root <- trajectory_df$CID == root_cid
  trajectory_df[, c(
    "CID", "MacroGroup", "UMAP_1", "UMAP_2",
    "DC1", "DC2", "DC3", "Pseudotime", "Is_root"
  )]
}

trajectory_coordinates <- bind_rows(lapply(
  levels(global_partition$MacroGroup),
  run_macrogroup_trajectory
))
write.csv(
  trajectory_coordinates,
  file.path(output_dir, "MacroGroup_Pseudotime_Coordinates.csv"),
  row.names = FALSE
)
