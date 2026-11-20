#!/usr/bin/env Rscript
# Rscript --vanilla spotclean09f.R INPUT_DIR OUTPUT_DIR
# Rscript --vanilla spotclean09f.R --smoke OUTPUT_DIR
# SpotClean's image coordinates can be expressed in micrometres. On the
# native grid 1 array step = 2 um. No slide image is an input to the fit.

args <- commandArgs(trailingOnly = TRUE)
regional <- length(args) == 3L && identical(args[3], "--09g-shared-plan")
if ((!regional && length(args) != 2L) || (regional && identical(args[1], "--smoke"))) {
    stop("Usage: spotclean09f.R INPUT_DIR|--smoke OUTPUT_DIR [--09g-shared-plan]")
}
minor <- paste(R.version$major, strsplit(R.version$minor, ".", fixed = TRUE)[[1]][1], sep = ".")
lib <- path.expand(Sys.getenv("SPOTCLEAN_R_LIB", file.path("~/.local/share/ct26/R", minor, "spotclean")))
.libPaths(c(lib, .libPaths()))
if (!requireNamespace("SpotClean", quietly = TRUE)) {
    stop("SpotClean could not load in ", R.home(), ". Run setup_09f_spotclean.R with this Rscript.")
}

write_tsv <- function(tab, path) {
    write.table(tab, path, quote = FALSE, row.names = FALSE, sep = "\t", na = "NA")
}
read_tsv <- function(path) {
    # Gene symbols such as NA and purely numeric barcodes are valid IDs.
    read.delim(path, check.names = FALSE, colClasses = "character",
               na.strings = character(), quote = "", comment.char = "")
}

peak_rss_mib <- function() {
    if (!file.exists("/proc/self/status")) return(NA_real_)
    lines <- readLines("/proc/self/status", warn = FALSE)
    mark <- grep("^VmHWM:", lines, value = TRUE)
    if (!length(mark)) return(NA_real_)
    as.numeric(strsplit(trimws(mark[1]), "[[:space:]]+")[[1]][2]) / 1024
}

# Deterministic miniature edge field; it goes through the exact same SpotClean
# code path as the biological ROI and includes an empty tissue/background bin.
smoke_inputs <- function() {
    set.seed(526)
    pos <- expand.grid(array_row = 0:7, array_col = 0:7)
    pos$barcode <- sprintf("smoke_%02d", seq_len(nrow(pos)))
    pos$x_um <- 2 * pos$array_row
    pos$y_um <- 2 * pos$array_col
    pos$in_tissue <- as.integer(pos$array_col < 5)
    genes <- data.frame(gene = c("Cd4", "Cd8a", "C1qa", paste0("Ref", 1:20), "BgOnly", "Absent"),
                        target = c(rep(1L, 3), rep(0L, 20), 1L, 1L))
    rate <- ifelse(pos$in_tissue == 1, 2.5 + pos$array_row / 6, 0.5)
    x <- matrix(rpois(23 * nrow(pos), rep(rate, each = 23)), nrow = 23)
    x <- rbind(x, as.integer(pos$in_tissue == 0), integer(nrow(pos)))
    x[, c(1L, ncol(x))] <- 0
    list(counts = Matrix::Matrix(x, sparse = TRUE), genes = genes, pos = pos)
}

run_fit <- function(dat, out, smoke) {
    pos <- dat$pos
    genes <- dat$genes
    counts <- methods::as(dat$counts, "CsparseMatrix")
    if (!all(c("in_tissue", "barcode", "x_um", "y_um") %in% names(pos)) ||
        !"gene" %in% names(genes)) stop("Missing columns in positions.tsv or genes.tsv.")
    if (!"target" %in% names(genes)) genes$target <- 0L
    if (!all(genes$target %in% c("0", "1")) || !all(pos$in_tissue %in% c("0", "1"))) {
        stop("target and in_tissue must use 0/1 labels.")
    }
    genes$target <- as.integer(genes$target)
    tissue <- as.integer(pos$in_tissue) == 1L
    if (!identical(dim(counts), c(nrow(genes), nrow(pos))) ||
        anyNA(genes$gene) || anyNA(pos$barcode) || any(!nzchar(genes$gene)) ||
        any(!nzchar(pos$barcode)) || anyDuplicated(genes$gene) || anyDuplicated(pos$barcode)) {
        stop("Matrix dimensions and unique TSV IDs must agree.")
    }
    limit <- if (regional) 4096L else 2500L
    if (nrow(pos) > limit) stop("Native window exceeds this pilot's ", limit, "-position limit.")
    if (sum(tissue) < 2L || sum(!tissue) < 2L) stop("SpotClean needs both tissue and background positions.")
    if (any(!is.finite(counts@x)) || any(counts@x < 0 | counts@x != floor(counts@x))) {
        stop("SpotClean requires nonnegative raw integer counts.")
    }
    dimnames(counts) <- list(genes$gene, pos$barcode)

    if (all(c("array_row", "array_col") %in% names(pos))) {
        # The image may be rotated. Use native square geometry for the model
        # and preserve anatomical coordinates in the original positions.tsv.
        row <- as.numeric(pos$array_row)
        col <- as.numeric(pos$array_col)
        if (any(!is.finite(c(row, col))) || any(c(row, col) != round(c(row, col)))) {
            stop("Native array indices must be finite integers.")
        }
        geometry <- "2um_array"
    } else {
        row <- as.numeric(pos$y_um) / 2
        col <- as.numeric(pos$x_um) / 2
        geometry <- "physical_coordinates_um"
    }
    if (any(!is.finite(c(row, col))) || anyDuplicated(data.frame(row, col)) ||
        length(unique(col)) < 2L || length(unique(row)) < 2L) stop("Invalid 2D HD geometry.")
    slide <- data.frame(barcode = pos$barcode, tissue = as.integer(tissue),
                        row = row, col = col, imagerow = 2 * row, imagecol = 2 * col)
    tissue_total <- as.numeric(Matrix::rowSums(counts[, tissue, drop = FALSE]))
    bg_total <- as.numeric(Matrix::rowSums(counts[, !tissue, drop = FALSE]))
    positive <- which(tissue_total > 0)
    # SpotClean estimates shared bleeding parameters from the EM gene set.
    # Include abundant references as well as every expressed target.
    ordered <- positive[order(-tissue_total[positive], genes$gene[positive])]
    fit_idx <- union(head(ordered, 200L), which(genes$target == 1L & tissue_total > 0))
    if (regional) {
        plan <- read_tsv(file.path(out, "shared_plan.tsv"))
        if (!"gene" %in% names(plan) || anyDuplicated(plan$gene) || !length(plan$gene)) {
            stop("Invalid 09G spatial gene panel.")
        }
        fit_idx <- match(plan$gene, genes$gene)
        if (anyNA(fit_idx) || any(tissue_total[fit_idx] <= 0) ||
            !all(which(genes$target == 1L & tissue_total > 0) %in% fit_idx) ||
            length(fit_idx) > 200L + sum(genes$target == 1L)) {
            stop("09G spatial panel differs from the expressed targets and reference gene plan.")
        }
    }
    if (length(fit_idx) < 2L ||
        sum(Matrix::colSums(counts[fit_idx, tissue, drop = FALSE]) > 0) < 2L) {
        stop("Insufficient positive tissue sources or expressed genes for SpotClean.")
    }
    if (sum(bg_total[fit_idx]) == 0) {
        stop("No background counts for the spatially fitted genes; a bleed rate cannot be estimated.")
    }
    all_bcs <- pos$barcode
    zero_bin <- Matrix::colSums(counts) == 0
    gene_status <- data.frame(gene = genes$gene, target = genes$target,
                              raw_tissue_umi = tissue_total, raw_background_umi = bg_total,
                              status = ifelse(tissue_total > 0, "scaled_only",
                                              ifelse(bg_total > 0, "background_only", "undetected")))
    gene_status$status[fit_idx] <- "spatial_em"
    write_tsv(gene_status, file.path(out, "gene_plan.tsv"))
    rad_um <- if (smoke) c(4, 8) else c(4, 10, 20)
    maxit <- if (smoke) 5L else 30L
    message("SpotClean ", packageVersion("SpotClean"), "; ", nrow(pos), " positions, ",
            sum(tissue), " tissue, ", sum(zero_bin), " zero-UMI positions, ", length(fit_idx),
            " genes for the spatial EM; candidate radii [um] ", paste(rad_um, collapse = ", "))
    slide_obj <- SpotClean::createSlide(counts, list(slide = slide), gene_cutoff = 0, verbose = FALSE)
    if (!identical(colnames(slide_obj), all_bcs)) stop("createSlide changed the native position ordering.")
    started <- proc.time()[["elapsed"]]
    corrected <- SpotClean::spotclean(slide_obj, gene_keep = genes$gene[fit_idx],
                                      candidate_radius = rad_um / 2, maxit = maxit, tol = 1,
                                      kernel = "gaussian", verbose = TRUE)
    elapsed <- proc.time()[["elapsed"]] - started
    result <- methods::as(SummarizedExperiment::assay(corrected, "decont"), "CsparseMatrix")
    expected_bcs <- all_bcs[tissue]
    if (!setequal(colnames(result), expected_bcs) || anyDuplicated(colnames(result)) ||
        any(!is.finite(result@x)) || any(result@x < 0) || !nrow(result) ||
        !all(rownames(result) %in% genes$gene[positive])) stop("Invalid corrected assay.")
    result <- result[, expected_bcs, drop = FALSE]
    meta <- S4Vectors::metadata(corrected)
    if (!setequal(meta$decontaminated_genes, genes$gene[fit_idx])) {
        stop("SpotClean's list of spatially fitted genes differs from the submitted gene set.")
    }
    gene_status$in_output <- as.integer(genes$gene %in% rownames(result))
    gone <- tissue_total > 0 & gene_status$in_output == 0
    gene_status$status[gone] <- "zero_after_fit"
    fit <- data.frame(
        run_type = if (smoke) "synthetic" else "native_roi",
        R_version = as.character(getRversion()), spotclean_version = as.character(packageVersion("SpotClean")),
        positions = nrow(pos), tissue_positions = sum(tissue), background_positions = sum(!tissue),
        zero_tissue_positions = sum(tissue & zero_bin), zero_background_positions = sum(!tissue & zero_bin),
        genes_spatial_em = length(fit_idx), genes_scaled_only = sum(gene_status$status == "scaled_only"),
        geometry = geometry, candidate_radii_um = paste(rad_um, collapse = ","),
        bleeding_rate = meta$bleeding_rate, distal_rate = meta$distal_rate,
        radius_um = 2 * meta$contamination_radius, arc_score = as.numeric(meta$ARC_score),
        em_iterations = length(meta$loglh), maxit = maxit, tol = 1, elapsed_seconds = elapsed,
        process_peak_rss_mib = peak_rss_mib())
    if (!all(is.finite(c(fit$bleeding_rate, fit$distal_rate, fit$radius_um, meta$loglh)))) {
        stop("Fit has non-finite parameters or likelihoods.")
    }
    if (smoke && (sum(gone) > 0L || !all(c("background_only", "undetected") %in% gene_status$status) ||
                  fit$zero_tissue_positions < 1L || fit$zero_background_positions < 1L)) {
        stop("Synthetic fit failed its coverage checks.")
    }
    Matrix::writeMM(result, file.path(out, "decont.mtx"))
    write_tsv(data.frame(gene = rownames(result)), file.path(out, "decont_genes.tsv"))
    write_tsv(data.frame(barcode = colnames(result)), file.path(out, "decont_positions.tsv"))
    write_tsv(gene_status, file.path(out, "gene_status.tsv"))
    write_tsv(data.frame(iteration = seq_along(meta$loglh), loglik = meta$loglh),
              file.path(out, "likelihood.tsv"))
    writeLines(capture.output(sessionInfo()), file.path(out, "sessionInfo.txt"))
    # Written last, after dimension, ordering and finiteness checks.
    write_tsv(fit, file.path(out, "fit.tsv"))
    message("Completed SpotClean in ", round(elapsed, 2), " seconds. Peak R process RSS ",
            round(fit$process_peak_rss_mib, 1), " MiB.")
}

out <- args[2]
dir.create(out, recursive = TRUE, showWarnings = FALSE)
if (!dir.exists(out)) stop("Cannot create output directory: ", out)
# A completed fit is immutable. The notebook allocates a fresh directory per run.
if (file.exists(file.path(out, "fit.tsv"))) stop("A completed fit exists at ", out, ". Use a fresh output directory.")
smoke <- identical(args[1], "--smoke")
dat <- if (smoke) smoke_inputs() else list(
    counts = Matrix::readMM(file.path(args[1], "counts.mtx")),
    genes = read_tsv(file.path(args[1], "genes.tsv")),
    pos = read_tsv(file.path(args[1], "positions.tsv")))
run_fit(dat, out, smoke)
