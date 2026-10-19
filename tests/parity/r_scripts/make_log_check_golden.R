# Golden outputs for the log-scale parity test.
#
# Reads every CSV in tests/parity/data/log_check/inputs/ and writes R
# MetaIntegrator's own decisions to tests/parity/data/log_check/expected/.
# tests/parity/test_log_check_parity.py then compares the Python port in
# src/preprocessing/utils/log_scale.py against these files.
#
# Usage, from the repository root:
#
#   Rscript tests/parity/r_scripts/make_log_check_golden.R
#
# Needs R 4.1 or newer and the MetaIntegrator package. The decision comes
# from the package itself (MetaIntegrator:::.GEM_log_check). The score and
# correlations come from `log_check_details()` below, a copy of the same
# function that also returns its intermediate values. The script stops if
# the copy and the package ever disagree on a decision.

suppressPackageStartupMessages(library(MetaIntegrator))

root <- "tests/parity/data/log_check"
in_dir <- file.path(root, "inputs")
out_dir <- file.path(root, "expected")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# A copy of MetaIntegrator's .GEM_log_check (R/getGEOData.R) that also
# returns the score and the three QQ-plot correlations.
log_check_details <- function(GEM) {
  goodCols <- which(sapply(1:ncol(GEM$expr),
                           function(i) !stats::sd(GEM$expr[, i], na.rm = T) == 0))
  ref_p <- goodCols[which(sapply(goodCols,
                                 function(i) !all(is.na(GEM$expr[, i]))))[1]]
  obj_real <- stats::qqnorm(GEM$expr[, ref_p], plot.it = FALSE)
  obj_exp  <- stats::qqnorm(exp(GEM$expr[, ref_p]), plot.it = FALSE)
  obj_log  <- stats::qqnorm(log(abs(GEM$expr[, ref_p]) + 0.00001), plot.it = FALSE)
  obj_exp$x[which(is.infinite(obj_exp$x))] <- NA
  obj_exp$y[which(is.infinite(obj_exp$y))] <- NA
  cor_real <- stats::cor(obj_real$x, obj_real$y, use = "pairwise.complete")
  cor_exp  <- stats::cor(obj_exp$x, obj_exp$y, use = "pairwise.complete")
  cor_log  <- stats::cor(obj_log$x, obj_log$y, use = "pairwise.complete")
  score <- NA
  if (!all(is.na(cor_exp))) {
    score <- log2(abs(cor_real**2 - cor_log**2) / abs(cor_real**2 - cor_exp**2))
    branch <- if (is.nan(score)) "check_failed" else "qq_score"
    is_log <- if (is.nan(score)) TRUE else score <= 0
  } else {
    branch <- "qq_fallback_real_vs_log"
    is_log <- cor_real > cor_log
  }
  list(is_log = is_log, branch = branch, score = score,
       cor_real = cor_real, cor_log = cor_log, cor_exp = cor_exp,
       reference_column = colnames(GEM$expr)[ref_p])
}

# getGEOData()'s transform (R/getGEOData.R), applied when not log scale.
r_transform <- function(expr) {
  min_value <- min(expr, na.rm = T)
  if (min_value < 0) {
    expr <- expr - min_value + 1
  }
  log2(expr)
}

files <- sort(list.files(in_dir, pattern = "\\.csv$", full.names = TRUE))
rows <- list()
for (path in files) {
  name <- sub("\\.csv$", "", basename(path))
  expr <- as.matrix(read.csv(path, row.names = 1, check.names = FALSE))
  gem <- list(expr = expr)

  package_says <- suppressWarnings(MetaIntegrator:::.GEM_log_check(gem))
  # getGEOData() transforms only when the result == FALSE. A warning
  # string (the "check failed" case) is not FALSE.
  package_transforms <- isTRUE(package_says == FALSE)
  details <- suppressWarnings(log_check_details(gem))
  if (package_transforms == isTRUE(details$is_log)) {
    stop(sprintf("%s: the copy and the package disagree", name))
  }

  rows[[name]] <- data.frame(
    case = name,
    log2_applied = package_transforms,
    decided_by = details$branch,
    score = details$score,
    cor_real = details$cor_real,
    cor_log = details$cor_log,
    cor_exp = details$cor_exp,
    reference_column = details$reference_column,
    r_version = R.version.string,
    metaintegrator_version = as.character(packageVersion("MetaIntegrator")),
    stringsAsFactors = FALSE
  )

  if (package_transforms) {
    out <- r_transform(expr)
    write.csv(out, file.path(out_dir, paste0(name, "_transformed.csv")))
  }
}

decisions <- do.call(rbind, rows)
write.csv(decisions, file.path(out_dir, "decisions.csv"), row.names = FALSE)
cat(sprintf("Wrote %d decisions to %s\n", nrow(decisions), out_dir))
