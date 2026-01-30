library(L1pack)
library(jsonlite)
library(dplyr)
library(stringr)

# Get command-line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 0) {
  stop("No experiment names provided. Usage: Rscript script.R experiment1 experiment2 ...")
}

# Base directories
data_base_dir <- "../../data"
out_base_dir <- "../../out"

# Process each experiment
for (experiment in args) {
  data_dir <- file.path(data_base_dir, experiment)
  out_file <- file.path(out_base_dir, paste0(experiment, "_R.csv"))
  
  # Check if directory exists
  if (!dir.exists(data_dir)) {
    warning(paste("Directory not found:", data_dir))
    next
  }

  # List all JSON files in subdirectories
  json_files <- list.files(path = data_dir, pattern = "*.json", recursive = TRUE, full.names = TRUE)

  # Initialize results list
  results <- list()

  # Process each JSON file
  for (file in json_files) {
    print(paste("Processing file:", file))

    # Extract n from parent directory name like 'n10000000'
    n_dir <- basename(dirname(file))
    n_match <- str_match(n_dir, "^n(\\d+)$")

    if (is.na(n_match[1, 2])) {
      warning(paste("Could not extract n from directory name:", file))
      next
    }

    n <- as.integer(n_match[1, 2])

    if (n <= 100000) {
      # Read and parse JSON
      # Wrap in tryCatch to prevent script crash on bad JSON
      tryCatch({
        data <- fromJSON(file)
        
        # Extract uid
        uid <- if (!is.null(data$uid)) as.integer(data$uid) else as.integer(NA)

        # Extract xs and ys
        xs <- as.matrix(data$xs)
        ys <- as.vector(data$ys)

        # Ensure correct dimensions
        if (length(xs) != length(ys)) {
          warning(paste("Skipping file due to mismatched dimensions:", file))
          next
        }

        # Benchmark l1fit
        start_time <- Sys.time()
        fit <- l1fit(xs, ys, intercept=TRUE)
        end_time <- Sys.time()
        
        runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))

        # Compute objective value (sum of absolute residuals)
        objective <- fit$minimum

        # Store result with objective and UID
        results[[file]] <- data.frame(
          n_samples = n, 
          uid = uid, 
          runtime = runtime, 
          objective = objective
        )
      }, error = function(e) {
        warning(paste("Error processing file:", file, "\nMessage:", e$message))
      })
      
    } else {
      # Skip parsing, record NA
      results[[file]] <- data.frame(
        n_samples = n, 
        uid = as.integer(NA), 
        runtime = as.numeric(NA), 
        objective = as.numeric(NA)
      )
    }
  }

  # Combine results and save as CSV
  if (length(results) > 0) {
    results_df <- bind_rows(results)
    write.csv(results_df, out_file, row.names = FALSE)
    print(paste("Benchmarking complete for", experiment, "Results saved to", out_file))
  } else {
    print(paste("No valid data found for", experiment, "No results saved."))
  }
}
