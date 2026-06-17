#!/usr/bin/env Rscript

# Robust R Data Converter - Enhanced to handle complex nested structures

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  cat("Usage: Rscript robust_converter.R input.RData [output_directory] [data_type]\n")
  cat("  data_type: 'input' (default) or 'combined' (for combineCOCOoutput results)\n")
  quit(status = 1)
}

input_file <- args[1]
output_dir <- if (length(args) >= 2) args[2] else "output_csv"
data_type <- if (length(args) >= 3) args[3] else "input"

# Check if input file exists
if (!file.exists(input_file)) {
  cat("Error: Input file", input_file, "does not exist!\n")
  quit(status = 1)
}

# Create output directory
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  cat("Created output directory:", output_dir, "\n")
}

cat("Loading R data from:", input_file, "\n")

# Capture objects before loading
objects_before <- ls()

# Load the R data file
load(input_file)
objects_loaded <- setdiff(ls(), objects_before)

cat("📊 Objects found:", length(objects_loaded), "\n")
cat("📋 Object names:", paste(objects_loaded, collapse = ", "), "\n")

converted_count <- 0
failed_count <- 0


# Enhanced safe convert function that handles complex nested structures
enhanced_safe_convert <- function(data, filename, description = "", max_depth = 0) {
  tryCatch({
    if (is.data.frame(data)) {
      write.csv(data, filename, row.names = TRUE)
      cat("   ✅", description, "data.frame saved:", basename(filename),
          "(", nrow(data), "×", ncol(data), ")\n")
      return(TRUE)

    } else if (is.matrix(data)) {
      write.csv(as.data.frame(data), filename, row.names = TRUE)
      cat("   ✅", description, "matrix saved:", basename(filename),
          "(", nrow(data), "×", ncol(data), ")\n")
      return(TRUE)

    } else if (is.vector(data) || length(data) >= 1) {
      # Handle all vector types including single values
      if (is.character(data)) {
        if (length(data) == 1) {
          # Single character string - save as text file
          txt_filename <- gsub("\\.csv$", ".txt", filename)
          writeLines(as.character(data), txt_filename)
          cat("   ✅", description, "text saved:", basename(txt_filename), "\n")
          return(TRUE)
        } else {
          # Multiple character values
          df <- data.frame(
            index = seq_along(data),
            value = as.character(data),
            stringsAsFactors = FALSE
          )
          write.csv(df, filename, row.names = FALSE)
          cat("   ✅", description, "character vector saved:", basename(filename), "(length:", length(data), ")\n")
          return(TRUE)
        }
      } else if (is.numeric(data) || is.integer(data)) {
        # Numeric values (single or multiple)
        df <- data.frame(
          index = if (length(data) > 1) seq_along(data) else 1,
          value = as.numeric(data)
        )
        write.csv(df, filename, row.names = FALSE)
        cat("   ✅", description, "numeric saved:", basename(filename), "(length:", length(data), ")\n")
        return(TRUE)

      } else if (is.logical(data)) {
        # Logical values (TRUE/FALSE)
        df <- data.frame(
          index = if (length(data) > 1) seq_along(data) else 1,
          value = as.logical(data)
        )
        write.csv(df, filename, row.names = FALSE)
        cat("   ✅", description, "logical saved:", basename(filename), "(length:", length(data), ")\n")
        return(TRUE)

      } else if (inherits(data, c("POSIXct", "POSIXt", "Date"))) {
        # Date/time objects
        df <- data.frame(
          index = if (length(data) > 1) seq_along(data) else 1,
          datetime = as.character(data)
        )
        write.csv(df, filename, row.names = FALSE)
        cat("   ✅", description, "datetime saved:", basename(filename), "\n")
        return(TRUE)

      } else {
        # Other vector types - try to convert to character
        df <- data.frame(
          index = if (length(data) > 1) seq_along(data) else 1,
          value = as.character(data)
        )
        write.csv(df, filename, row.names = FALSE)
        cat("   ✅", description, "converted vector saved:", basename(filename), "(length:", length(data), ")\n")
        return(TRUE)
      }

    } else if (is.list(data) && max_depth > 0) {
      # Recursively handle nested lists
      return(handle_nested_list(data, filename, description, max_depth - 1))

    } else {
      cat("   ⚠️ ", description, "- unsupported type:", paste(class(data), collapse = "/"),
          "(length:", length(data), ")\n")
      return(FALSE)
    }
  }, error = function(e) {
    cat("   ❌", description, "conversion failed:", e$message, "\n")
    return(FALSE)
  })
}


# Function to handle deeply nested list structures
handle_nested_list <- function(list_data, base_filename, description, max_depth) {
  if (!is.list(list_data) || max_depth <= 0) {
    return(FALSE)
  }

  local_success_count <- 0
  base_name <- gsub("\\.csv$", "", base_filename)

  # Try different strategies for list conversion

  # Strategy 1: Check if all elements are compatible for data.frame conversion
  if (try_dataframe_conversion(list_data, base_filename, description)) {
    return(TRUE)
  }

  # Strategy 2: Save each list element separately
  for (i in seq_along(list_data)) {
    element_name <- if (!is.null(names(list_data))) names(list_data)[i] else paste0("element_", i)
    element_data <- list_data[[i]]
    element_filename <- paste0(base_name, "_", element_name, ".csv")
    element_description <- paste(description, "-", element_name)

    if (enhanced_safe_convert(element_data, element_filename, element_description, max_depth)) {
      local_success_count <- local_success_count + 1
    }
  }

  return(local_success_count > 0)
}

# Try to convert list to data frame if elements are compatible
try_dataframe_conversion <- function(list_data, filename, description) {
  tryCatch({
    # Check if all elements have the same length (can become columns)
    lengths <- sapply(list_data, length)

    if (length(unique(lengths)) == 1 && all(lengths > 0)) {
      # All elements have same length - try to make data frame
      df <- data.frame(list_data, stringsAsFactors = FALSE)
      write.csv(df, filename, row.names = FALSE)
      cat("   ✅", description, "list converted to data.frame:", basename(filename),
          "(", nrow(df), "×", ncol(df), ")\n")
      return(TRUE)
    }

    # Check if it's a list of data frames that can be row-bound
    if (all(sapply(list_data, is.data.frame))) {
      # Check if all data frames have the same columns
      col_names <- lapply(list_data, colnames)
      if (length(unique(lapply(col_names, sort))) == 1) {
        df <- do.call(rbind, list_data)
        write.csv(df, filename, row.names = TRUE)
        cat("   ✅", description, "list of data.frames combined:", basename(filename),
            "(", nrow(df), "×", ncol(df), ")\n")
        return(TRUE)
      }
    }

    return(FALSE)
  }, error = function(e) {
    return(FALSE)
  })
}

# Function to process any complex object recursively
process_complex_object <- function(obj_name, obj_data, current_path = "") {
  full_name <- if (current_path == "") obj_name else paste(current_path, obj_name, sep = "_")

  cat("\n📋 Processing:", full_name, "\n")
  cat("   Type:", paste(class(obj_data), collapse = ", "), "\n")

  local_converted <- 0

  if (is.list(obj_data) && !is.data.frame(obj_data)) {
    cat("   Structure: List with", length(obj_data), "elements\n")
    if (!is.null(names(obj_data))) {
      cat("   Element names:", paste(names(obj_data), collapse = ", "), "\n")
    }

    # Handle each list element
    for (i in seq_along(obj_data)) {
      element_name <- if (!is.null(names(obj_data))) names(obj_data)[i] else paste0("element_", i)
      element_data <- obj_data[[i]]

      cat("\n  📂 Processing element:", element_name, "\n")
      cat("     Type:", paste(class(element_data), collapse = ", "), "\n")

      element_path <- paste(full_name, element_name, sep = "_")
      filename <- file.path(output_dir, paste0(element_path, ".csv"))
      description <- paste("Element", element_name)

      # Determine processing strategy based on data type and complexity
      if (is.list(element_data) && !is.data.frame(element_data)) {
        # Further nested structure
        if (length(element_data) > 0) {
          cat("     Sub-structure: List with", length(element_data), "sub-elements\n")
          if (!is.null(names(element_data))) {
            cat("     Sub-element names:", paste(names(element_data), collapse = ", "), "\n")
          }

          # Recursively process this nested structure
          nested_converted <- process_complex_object(element_name, element_data, full_name)
          local_converted <- local_converted + nested_converted
        } else {
          cat("     ⚠️ Empty list, skipping\n")
        }
      } else {
        # Try to convert this element directly
        if (enhanced_safe_convert(element_data, filename, description, max_depth = 2)) {
          local_converted <- local_converted + 1
        } else {
          failed_count <<- failed_count + 1
        }
      }
    }

  } else {
    # Simple object - convert directly
    filename <- file.path(output_dir, paste0(full_name, ".csv"))
    if (enhanced_safe_convert(obj_data, filename, paste("Object", full_name), max_depth = 2)) {
      local_converted <- local_converted + 1
    } else {
      failed_count <<- failed_count + 1
    }
  }

  return(local_converted)
}

# Process each loaded object
for (obj_name in objects_loaded) {
  obj_data <- get(obj_name)

  # Skip script variables
  if (obj_name %in% c("objects_before", "args", "input_file", "output_dir", "data_type")) {
    next
  }

  converted_count <- converted_count + process_complex_object(obj_name, obj_data)
}

# Create a summary file
summary_file <- file.path(output_dir, "conversion_summary.txt")
tryCatch({
  sink(summary_file)
  cat("R Data Conversion Summary\n")
  cat("========================\n\n")
  cat("Input file:", input_file, "\n")
  cat("Processing time:", as.character(Sys.time()), "\n")
  cat("Successfully converted:", converted_count, "files\n")
  cat("Failed conversions:", failed_count, "items\n\n")

  cat("Source objects:\n")
  for (obj_name in objects_loaded) {
    if (obj_name %in% c("objects_before", "args", "input_file", "output_dir", "data_type")) next
    obj_data <- get(obj_name)
    cat("- ", obj_name, " (", paste(class(obj_data), collapse = "/"), ")\n")
  }

  cat("\nOutput files:\n")
  output_files <- list.files(output_dir, recursive = TRUE)
  for (file in output_files) {
    if (file != "conversion_summary.txt") {
      cat("- ", file, "\n")
    }
  }
  sink()
}, error = function(e) {
  cat("Could not create summary file\n")
})

# Final summary
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("📊 CONVERSION SUMMARY:\n")
cat("   ✅ Successfully converted:", converted_count, "files\n")
cat("   ❌ Failed to convert:", failed_count, "items\n")
cat("   📁 Output directory:", output_dir, "\n")
cat("   📄 Summary file:", summary_file, "\n")

if (converted_count > 0) {
  cat("\n🎉 Conversion completed!\n")
  cat("📋 Files created:\n")

  # List the created files
  csv_files <- list.files(output_dir, pattern = "\\.(csv|txt)$", recursive = TRUE)
  for (file in csv_files) {
    cat("   -", file, "\n")
  }

  if (failed_count > 0) {
    cat("\n💡 Some items couldn't be converted (usually complex nested structures or incompatible data types)\n")
    cat("   Check the output files to see what was successfully converted.\n")
  }
} else {
  cat("\n❌ No files were converted successfully.\n")
  cat("   The data structure might be too complex or contain unsupported data types.\n")
}
