# 04D final hierarchical assignments
# Core path entry: resolve project root from Rscript, independent of caller cwd.
.atlas_file <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(.atlas_file) != 1L) stop("Run with Rscript.")
.atlas_dir <- dirname(normalizePath(sub("^--file=", "", .atlas_file), mustWork = TRUE))
setwd(normalizePath(file.path(.atlas_dir, ".."), mustWork = TRUE))
source(file.path("configs", "paths.R"))


args <- commandArgs(trailingOnly = TRUE)
if (!length(args) %in% c(0L, 3L)) {
  stop("Supply a MacroGroup partition CSV and the two MacroGroup subcluster CSVs.")
}
input_paths <- if (length(args)) args else c(
  frozen_partition,
  file.path(frozen_dir, "04A", c(
    "MacroGroup_1_CompoundClusters.csv", "MacroGroup_2_CompoundClusters.csv"
  ))
)
global_umap_path <- input_paths[[1L]]
mg1_path <- input_paths[[2L]]
mg2_path <- input_paths[[3L]]
output_dir <- file.path(run_dir, "04D")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

read_required_csv <- function(path, required_columns) {
  if (!file.exists(path)) stop("Required input is missing: ", path)
  x <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  missing <- setdiff(required_columns, names(x))
  if (length(missing)) stop("Missing required column(s) in ", path, ": ", paste(missing, collapse = ", "))
  x
}

global_umap <- read_required_csv(global_umap_path, c("CID", "MacroGroup", "UMAP1", "UMAP2"))
mg1 <- read_required_csv(mg1_path, c("CID", "CompoundCluster"))
mg2 <- read_required_csv(mg2_path, c("CID", "CompoundCluster"))

global_umap$CID <- as.character(global_umap$CID)
mg1$CID <- as.character(mg1$CID)
mg2$CID <- as.character(mg2$CID)

assert_unique_cid <- function(x, name) {
  duplicate_cids <- unique(x$CID[duplicated(x$CID)])
  if (length(duplicate_cids)) {
    stop(name, " contains duplicate CID(s): ", paste(duplicate_cids, collapse = ", "))
  }
}
assert_unique_cid(global_umap, "Global UMAP input")
assert_unique_cid(mg1, "MacroGroup 1 cluster input")
assert_unique_cid(mg2, "MacroGroup 2 cluster input")

normalise_c1_c2 <- function(x, name) {
  value <- trimws(as.character(x$CompoundCluster))
  is_valid <- value %in% c("Compound cluster 1", "Compound cluster 2")
  if (any(!is_valid)) {
    stop(name, " has non-C1/C2 CompoundCluster label(s): ", paste(unique(value[!is_valid]), collapse = ", "))
  }
  ifelse(value == "Compound cluster 1", "C1", "C2")
}
mg1$original_subcluster <- normalise_c1_c2(mg1, "MacroGroup 1 cluster input")
mg2$original_subcluster <- normalise_c1_c2(mg2, "MacroGroup 2 cluster input")

if (any(global_umap$CID %in% mg1$CID & global_umap$MacroGroup != 1)) {
  stop("MacroGroup conflict: a CID from the MacroGroup 1 source is not MacroGroup 1 in global UMAP.")
}
if (any(global_umap$CID %in% mg2$CID & global_umap$MacroGroup != 2)) {
  stop("MacroGroup conflict: a CID from the MacroGroup 2 source is not MacroGroup 2 in global UMAP.")
}
if (length(intersect(mg1$CID, mg2$CID))) {
  stop("CID(s) occur in both MacroGroup source files: ", paste(intersect(mg1$CID, mg2$CID), collapse = ", "))
}

subclusters <- rbind(
  data.frame(CID = mg1$CID, source_macrogroup = 1, original_subcluster = mg1$original_subcluster, stringsAsFactors = FALSE),
  data.frame(CID = mg2$CID, source_macrogroup = 2, original_subcluster = mg2$original_subcluster, stringsAsFactors = FALSE)
)

assignments <- merge(global_umap, subclusters, by = "CID", all = FALSE, sort = FALSE)
if (nrow(assignments) != nrow(subclusters)) {
  unmatched <- setdiff(subclusters$CID, global_umap$CID)
  stop("Unmatched CID(s) between subcluster sources and global UMAP: ", paste(unmatched, collapse = ", "))
}
if (any(assignments$MacroGroup != assignments$source_macrogroup)) {
  stop("MacroGroup conflict detected after merge.")
}

assignments$semantic_subcluster <- paste0("MG", assignments$MacroGroup, "_", assignments$original_subcluster)
semantic_to_id <- c(MG1_C1 = "1", MG1_C2 = "2", MG2_C1 = "3", MG2_C2 = "4")
assignments$cluster_id <- unname(semantic_to_id[assignments$semantic_subcluster])
if (anyNA(assignments$cluster_id)) stop("Unexpected semantic subcluster label after merge.")
assignments$cluster_id <- factor(assignments$cluster_id, levels = c("1", "2", "3", "4"))

expected_source <- c("1" = "MG1_C1", "2" = "MG1_C2", "3" = "MG2_C1", "4" = "MG2_C2")
for (cluster in names(expected_source)) {
  observed <- unique(assignments$semantic_subcluster[assignments$cluster_id == cluster])
  if (!identical(observed, expected_source[[cluster]])) {
    stop("Cluster mapping audit failed for cluster ", cluster, ".")
  }
}

cat("Global UMAP CID count:", nrow(global_umap), "\n")
cat("MacroGroup 1 cluster CID count:", nrow(mg1), "\n")
cat("MacroGroup 2 cluster CID count:", nrow(mg2), "\n")
cat("Successfully matched subcluster CID count:", nrow(assignments), "\n")
cat("Unmatched CID count: 0\n")
cat("Cluster counts:\n")
print(table(assignments$cluster_id))

assignments <- assignments[, c("CID", "UMAP1", "UMAP2", "MacroGroup", "original_subcluster", "semantic_subcluster", "cluster_id")]
write.csv(assignments, file.path(output_dir, "Subcluster_Assignments.csv"), row.names = FALSE)
label_mapping <- data.frame(
  cluster_id = c("1", "2", "3", "4"),
  subcluster = c("MG1_C1", "MG1_C2", "MG2_C1", "MG2_C2"),
  stringsAsFactors = FALSE
)
write.csv(label_mapping, file.path(output_dir, "subcluster_label_mapping.csv"), row.names = FALSE)
