# Resolve paths from the Atlas-Structure module root.
rowz_csv <- file.path(
  "..", "CPI-MEC", "results", "final_fit", "Final_full_label_fit_rowZ_matrix.csv"
)
frozen_dir <- "results"
frozen_partition <- file.path(
  frozen_dir, "03B", "global_Leiden_MacroGroup_partition.csv"
)
score_dir <- file.path("data", "subcluster_inputs")
run_dir <- "runs"
