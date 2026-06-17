#!/usr/bin/env Rscript

# Quick inspector for tinyMetaObject

input_file <- "data/reference/metaintegrator/rdata/tinyMetaObject.RData"

cat("🔍 Inspecting tinyMetaObject structure\n")
cat("=====================================\n")

# Load the file
load(input_file)

# Inspect the main object
cat("📊 Object name: tinyMetaObject\n")
cat("📊 Class:", paste(class(tinyMetaObject), collapse = ", "), "\n")
cat("📊 Type:", typeof(tinyMetaObject), "\n")

if (is.list(tinyMetaObject)) {
  cat("📊 List length:", length(tinyMetaObject), "\n")
  cat("📊 Element names:", paste(names(tinyMetaObject), collapse = ", "), "\n\n")

  # Show detailed structure
  cat("📋 DETAILED STRUCTURE:\n")
  str(tinyMetaObject, max.level = 3)

  cat("\n📂 ELEMENT DETAILS:\n")
  for (i in 1:min(5, length(tinyMetaObject))) {
    element_name <- names(tinyMetaObject)[i]
    element_data <- tinyMetaObject[[i]]

    cat("\n[", i, "] Element:", element_name, "\n")
    cat("    Class:", paste(class(element_data), collapse = ", "), "\n")

    if (is.data.frame(element_data)) {
      cat("    Dimensions:", nrow(element_data), "×", ncol(element_data), "\n")
      cat("    Columns:", paste(colnames(element_data)[1:min(5, ncol(element_data))], collapse = ", "))
      if (ncol(element_data) > 5) cat("...")
      cat("\n")
    } else if (is.matrix(element_data)) {
      cat("    Dimensions:", nrow(element_data), "×", ncol(element_data), "\n")
    } else if (is.vector(element_data)) {
      cat("    Length:", length(element_data), "\n")
      if (length(element_data) <= 10) {
        cat("    Values:", paste(element_data, collapse = ", "), "\n")
      } else {
        cat("    First values:", paste(head(element_data, 5), collapse = ", "), "...\n")
      }
    } else if (is.list(element_data)) {
      cat("    Sub-list length:", length(element_data), "\n")
      if (!is.null(names(element_data))) {
        cat("    Sub-names:", paste(names(element_data)[1:min(3, length(element_data))], collapse = ", "))
        if (length(element_data) > 3) cat("...")
        cat("\n")
      }
    }
  }

  if (length(tinyMetaObject) > 5) {
    cat("\n... and", length(tinyMetaObject) - 5, "more elements\n")
  }

} else if (is.data.frame(tinyMetaObject)) {
  cat("📊 Data frame dimensions:", nrow(tinyMetaObject), "×", ncol(tinyMetaObject), "\n")
  cat("📊 Columns:", paste(colnames(tinyMetaObject), collapse = ", "), "\n")
  cat("\n📋 STRUCTURE:\n")
  str(tinyMetaObject)
  cat("\n📋 FIRST FEW ROWS:\n")
  print(head(tinyMetaObject))

} else if (is.vector(tinyMetaObject)) {
  cat("📊 Vector length:", length(tinyMetaObject), "\n")
  cat("📊 Values:", paste(head(tinyMetaObject, 10), collapse = ", "))
  if (length(tinyMetaObject) > 10) cat("...")
  cat("\n")

} else {
  cat("\n📋 STRUCTURE:\n")
  str(tinyMetaObject)
}

cat("\n🎯 CONVERSION RECOMMENDATION:\n")
if (is.list(tinyMetaObject)) {
  cat("✅ This appears to be a complex list structure.\n")
  cat("💡 Use: Rscript enhanced_converter.R", input_file, "output_meta input\n")
} else if (is.data.frame(tinyMetaObject)) {
  cat("✅ This is a simple data frame.\n")
  cat("💡 Can be saved directly to CSV\n")
} else {
  cat("✅ This is a", class(tinyMetaObject)[1], "\n")
  cat("💡 May need special handling\n")
}