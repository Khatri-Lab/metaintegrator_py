#!/usr/bin/env Rscript

# Enhanced R Script to Convert Complex Nested RData Structures
# Now supports both input data and COCONUT combineCOCOoutput results

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  cat("Usage: Rscript enhanced_converter.R input.RData [output_directory] [data_type]\n")
  cat("  data_type: 'input' (default) or 'combined' (for combineCOCOoutput results)\n")
  cat("Examples:\n")
  cat("  Rscript enhanced_converter.R GSEs.test.RData data_txt input\n")
  cat("  Rscript enhanced_converter.R combined_results.RData output_txt combined\n")
  quit(status = 1)
}

input_file <- args[1]
output_dir <- if (length(args) >= 2) args[2] else "output_csv"
data_type <- if (length(args) >= 3) args[3] else "input"

# Validate data_type
if (!data_type %in% c("input", "combined")) {
  cat("Error: data_type must be 'input' or 'combined'\n")
  quit(status = 1)
}

# Check if input file exists
if (!file.exists(input_file)) {
  cat("Error: Input file", input_file, "does not exist!\n")
  quit(status = 1)
}

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  cat("Created output directory:", output_dir, "\n")
}

cat("Loading R data from:", input_file, "\n")
cat("Data type:", data_type, "\n")

# Capture objects before loading to exclude script variables
objects_before <- ls()

# Load the R data file
tryCatch({
  load(input_file)
  cat("✅ Successfully loaded R data file\n")
}, error = function(e) {
  cat("❌ Error loading file:", e$message, "\n")
  quit(status = 1)
})

# Get only the newly loaded objects (exclude script variables)
all_objects <- ls()
objects_loaded <- setdiff(all_objects, objects_before)

cat("📊 Objects found:", length(objects_loaded), "\n")
if (length(objects_loaded) > 0) {
  cat("📋 Object names:", paste(objects_loaded, collapse = ", "), "\n")
}

if (length(objects_loaded) == 0) {
  cat("❌ No new objects found in the R data file!\n")
  quit(status = 1)
}

converted_count <- 0
failed_count <- 0

# Function to safely convert and save data
safe_convert <- function(data, filename, description = "") {
  tryCatch({
    if (is.data.frame(data)) {
      write.csv(data, filename, row.names = TRUE)
      cat("   ✅", description, "data.frame saved to:", filename,
          "(", nrow(data), "×", ncol(data), ")\n")
      return(TRUE)
    } else if (is.matrix(data)) {
      df <- as.data.frame(data)
      write.csv(df, filename, row.names = TRUE)
      cat("   ✅", description, "matrix converted to data.frame and saved to:", filename,
          "(", nrow(data), "×", ncol(data), ")\n")
      return(TRUE)
    } else if (is.vector(data) && length(data) > 1) {
      df <- data.frame(value = data)
      write.csv(df, filename, row.names = FALSE)
      cat("   ✅", description, "vector saved to:", filename, "(length:", length(data), ")\n")
      return(TRUE)
    } else {
      cat("   ⚠️ ", description, "- unsupported type:", class(data)[1], "\n")
      return(FALSE)
    }
  }, error = function(e) {
    cat("   ❌", description, "conversion failed:", e$message, "\n")
    return(FALSE)
  })
}

# Function to handle combineCOCOoutput data structure
handle_combined_data <- function(obj_name, obj_data) {
  cat("\n📊 Processing combined COCONUT output:", obj_name, "\n")
  cat("   Type:", class(obj_data)[1], "\n")

  if (!is.list(obj_data)) {
    cat("   ❌ Expected list structure for combined data\n")
    return(0)
  }

  local_converted <- 0

  # Process each dataset in the combined results
  for (i in seq_along(obj_data)) {
    dataset <- obj_data[[i]]
    dataset_name <- paste0("dataset_", i)

    cat("\n  📂 Processing combined dataset", i, "\n")

    if (!is.list(dataset)) {
      cat("     ⚠️ Dataset", i, "is not a list, skipping\n")
      next
    }

    # Expected structure: genes, pheno, class
    expected_elements <- c("genes", "pheno", "class")
    available_elements <- intersect(names(dataset), expected_elements)

    if (length(available_elements) == 0) {
      cat("     ⚠️ No recognized elements (genes, pheno, class) in dataset", i, "\n")
      next
    }

    cat("     Available elements:", paste(available_elements, collapse = ", "), "\n")

    # Save each element
    for (element_name in available_elements) {
      element_data <- dataset[[element_name]]

      cat("\n    📄 Processing element:", element_name, "\n")
      cat("       Type:", class(element_data)[1], "\n")

      # Create filename for combined data
      filename <- file.path(output_dir, paste0(obj_name, "_", dataset_name, "_", element_name, ".csv"))
      description <- paste("Combined", element_name, "from dataset", i)

      # Special handling for class labels
      if (element_name == "class") {
        if (is.vector(element_data)) {
          df <- data.frame(sample_id = seq_along(element_data), class_label = element_data)
          write.csv(df, filename, row.names = FALSE)
          cat("       ✅", description, "vector saved to:", filename, "(length:", length(element_data), ")\n")
          local_converted <- local_converted + 1
        } else {
          cat("       ⚠️", description, "- expected vector for class labels\n")
        }
      } else {
        # Handle genes and pheno normally
        if (safe_convert(element_data, filename, description)) {
          local_converted <- local_converted + 1
        } else {
          failed_count <<- failed_count + 1
        }
      }
    }
  }

  # Also create a summary file for the combined results
  summary_file <- file.path(output_dir, paste0(obj_name, "_summary.txt"))

  tryCatch({
    sink(summary_file)
    cat("COCONUT Combined Results Summary\n")
    cat("================================\n\n")
    cat("Input file:", input_file, "\n")
    cat("Processing date:", Sys.time(), "\n")
    cat("Number of datasets:", length(obj_data), "\n\n")

    for (i in seq_along(obj_data)) {
      dataset <- obj_data[[i]]
      cat("Dataset", i, ":\n")

      if ("genes" %in% names(dataset)) {
        cat("  Genes: ", nrow(dataset$genes), "genes ×", ncol(dataset$genes), "samples\n")
      }
      if ("pheno" %in% names(dataset)) {
        cat("  Pheno: ", nrow(dataset$pheno), "samples ×", ncol(dataset$pheno), "attributes\n")
      }
      if ("class" %in% names(dataset)) {
        class_summary <- table(dataset$class)
        cat("  Classes: ", paste(names(class_summary), "=", class_summary, collapse = ", "), "\n")
      }
      cat("\n")
    }
    sink()

    cat("   ✅ Summary saved to:", summary_file, "\n")
    local_converted <- local_converted + 1

  }, error = function(e) {
    cat("   ⚠️ Could not create summary file:", e$message, "\n")
  })

  return(local_converted)
}

# Function to handle original nested input data structure
handle_input_data <- function(obj_name, obj_data) {
  cat("\n📋 Processing input object:", obj_name, "\n")
  cat("   Type:", class(obj_data)[1], "\n")

  local_converted <- 0

  if (is.list(obj_data) && !is.data.frame(obj_data)) {
    cat("   Structure: List with", length(obj_data), "elements\n")
    cat("   Element names:", paste(names(obj_data), collapse = ", "), "\n")

    # Handle nested list structure (like GSEs.test)
    for (element_name in names(obj_data)) {
      element_data <- obj_data[[element_name]]

      cat("\n  📂 Processing element:", element_name, "\n")
      cat("     Type:", class(element_data)[1], "\n")

      if (is.list(element_data) && !is.data.frame(element_data)) {
        cat("     Sub-structure: List with", length(element_data), "sub-elements\n")
        cat("     Sub-element names:", paste(names(element_data), collapse = ", "), "\n")

        # Handle sub-elements (like pheno and genes)
        for (sub_name in names(element_data)) {
          sub_data <- element_data[[sub_name]]

          cat("\n    📄 Processing sub-element:", sub_name, "\n")
          cat("       Type:", class(sub_data)[1], "\n")

          # Create descriptive filename
          filename <- file.path(output_dir, paste0(obj_name, "_", element_name, "_", sub_name, ".csv"))
          description <- paste("Sub-element", sub_name, "from", element_name)

          if (safe_convert(sub_data, filename, description)) {
            local_converted <- local_converted + 1
          } else {
            failed_count <<- failed_count + 1
          }
        }

      } else {
        # Direct element (not nested further)
        filename <- file.path(output_dir, paste0(obj_name, "_", element_name, ".csv"))
        description <- paste("Element", element_name)

        if (safe_convert(element_data, filename, description)) {
          local_converted <- local_converted + 1
        } else {
          failed_count <<- failed_count + 1
        }
      }
    }

  } else {
    # Simple object (data.frame, matrix, vector)
    filename <- file.path(output_dir, paste0(obj_name, ".csv"))

    if (safe_convert(obj_data, filename, paste("Object", obj_name))) {
      local_converted <- local_converted + 1
    } else {
      failed_count <<- failed_count + 1
    }
  }

  return(local_converted)
}

# Process each loaded object based on data type
for (obj_name in objects_loaded) {
  obj_data <- get(obj_name)

  if (data_type == "combined") {
    converted_count <- converted_count + handle_combined_data(obj_name, obj_data)
  } else {
    converted_count <- converted_count + handle_input_data(obj_name, obj_data)
  }
}

# Final summary
cat("\n", paste(rep("=", 50), collapse = ""), "\n")
cat("📊 CONVERSION SUMMARY:\n")
cat("   ✅ Successfully converted:", converted_count, "files\n")
cat("   ❌ Failed to convert:", failed_count, "items\n")
cat("   📁 Output directory:", output_dir, "\n")

if (converted_count > 0) {
  cat("\n🎉 Conversion completed successfully!\n")
  cat("📋 Files created:\n")

  # List the created files
  csv_files <- list.files(output_dir, pattern = "*", full.names = FALSE)
  for (file in csv_files) {
    cat("   -", file, "\n")
  }
} else {
  cat("\n❌ No files were converted!\n")
}

if (data_type == "combined") {
  cat("\n💡 COMBINED DATA STRUCTURE:\n")
  cat("   Each dataset now contains combined control + disease samples\n")
  cat("   Files are named: ObjectName_dataset_N_ElementType.csv\n")
  cat("   Elements: genes (expression), pheno (metadata), class (0=control, 1=disease)\n")
  cat("   Use class labels to identify control vs disease samples\n")
} else {
  cat("\n💡 INPUT DATA STRUCTURE:\n")
  cat("   - Each phenotype table (sample metadata)\n")
  cat("   - Each gene expression matrix\n")
  cat("   - Files are named: ObjectName_Dataset_DataType.csv\n")
  cat("   - Example: GSEs.test_GSE28750_pheno.csv, GSEs.test_GSE28750_genes.csv\n")
}
