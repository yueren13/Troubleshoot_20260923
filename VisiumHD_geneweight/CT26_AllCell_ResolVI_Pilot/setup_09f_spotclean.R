#!/usr/bin/env Rscript
# Run from any working directory, keeping this file alongside spotclean09f.R.
# Optional environment overrides: SPOTCLEAN_R_LIB, CT26_CRAN_REPO, CT26_BIOC_MIRROR.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 0L) stop("Usage: Rscript --vanilla setup_09f_spotclean.R")
minor <- paste(R.version$major, strsplit(R.version$minor, ".", fixed = TRUE)[[1]][1], sep = ".")
if (minor != "4.5") {
    stop("09F targets the existing Domino R 4.5.x. Rscript found ", getRversion(),
         " at ", R.home(), ". Set the notebook RSCRIPT path to the existing R 4.5.2 executable.")
}
lib <- path.expand(Sys.getenv("SPOTCLEAN_R_LIB", file.path("~/.local/share/ct26/R", minor, "spotclean")))
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
if (!dir.exists(lib) || file.access(lib, 2) != 0) stop("Cannot write the SpotClean library: ", lib)
lib <- normalizePath(lib)
.libPaths(c(lib, .libPaths()))
# Include the private library for nested R CMD INSTALL and the fresh-process smoke run.
Sys.setenv(R_LIBS = paste(.libPaths(), collapse = .Platform$path.sep), SPOTCLEAN_R_LIB = lib)
runner_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
runner <- file.path(dirname(normalizePath(sub("^--file=", "", runner_arg[1]))), "spotclean09f.R")
if (!file.exists(runner)) stop("Place spotclean09f.R alongside this setup file.")
state <- file.path(lib, "09f-setup-receipts")
dir.create(state, recursive = TRUE, showWarnings = FALSE)
receipt <- tempfile(pattern = "run_", tmpdir = state)
dir.create(receipt)
message("Setup receipt: ", receipt, "\nR ", getRversion(), " at ", R.home(),
        "\nFirst library: ", lib,
        "\nInherited libraries: ", paste(.libPaths()[-1], collapse = ", "))

started <- proc.time()[["elapsed"]]
cran <- Sys.getenv("CT26_CRAN_REPO", unname(getOption("repos")["CRAN"]))
if (length(cran) != 1L || is.na(cran) || !nzchar(cran) || cran == "@CRAN@") {
    cran <- "https://cloud.r-project.org"
}
options(repos = c(CRAN = cran), timeout = max(600, getOption("timeout")),
        Ncpus = 2L)
mirror <- Sys.getenv("CT26_BIOC_MIRROR")
if (nzchar(mirror)) options(BioC_mirror = mirror)

loadable <- function(pkg) requireNamespace(pkg, quietly = TRUE)
install_needed <- !loadable("SpotClean")
release <- NA_character_
if (install_needed) {
    if (!loadable("BiocManager")) {
        install.packages("BiocManager", lib = lib, repos = c(CRAN = cran),
                         dependencies = NA, Ncpus = 1L)
        if (!loadable("BiocManager")) stop("BiocManager installation failed. See the CRAN error above.")
    }
    # R 4.5 spans 3.21 and 3.22. Match a known release, defaulting to 3.22
    # on an installation where no BiocVersion package has been installed yet.
    release <- "3.22"
    if (loadable("BiocVersion")) {
        visible <- paste(head(strsplit(as.character(packageVersion("BiocVersion")), ".", fixed = TRUE)[[1]], 2L),
                         collapse = ".")
        if (visible %in% c("3.21", "3.22")) release <- visible
    }
    repos <- BiocManager::repositories(version = release)
    message("Installing SpotClean and its required dependencies from Bioconductor ", release,
            ". First-time builds can take longer than the biological pilot.")
    # Use the selected repositories without requesting an update of installed
    # packages or switching the shared BiocVersion installation to another release.
    install.packages("SpotClean", lib = lib, repos = repos, dependencies = NA, Ncpus = 2L)
    if (!loadable("SpotClean")) {
        failure <- tryCatch({ loadNamespace("SpotClean"); "" }, error = conditionMessage)
        stop("SpotClean could not load after installation: ", failure,
             ". The package-manager output above contains the first dependency or repository failure.")
    }
} else {
    message("Reusing SpotClean ", packageVersion("SpotClean"), " from ", find.package("SpotClean"))
}
writeLines(capture.output(sessionInfo()), file.path(receipt, "setup_sessionInfo.txt"))
smoke <- file.path(receipt, "smoke")
rscript <- file.path(R.home("bin"), "Rscript")
smoke_log <- file.path(receipt, "smoke.log")
status <- system2(rscript, c("--vanilla", shQuote(runner), "--smoke", shQuote(smoke)),
                  stdout = smoke_log, stderr = smoke_log)
if (file.exists(smoke_log)) cat(readLines(smoke_log, warn = FALSE), sep = "\n")
if (status != 0L || !file.exists(file.path(smoke, "fit.tsv"))) {
    stop("SpotClean smoke fit failed. Full log: ", smoke_log)
}
setup <- data.frame(r_version = as.character(getRversion()), rscript = rscript,
                    spotclean_version = as.character(packageVersion("SpotClean")),
                    package_path = find.package("SpotClean"), private_library = lib,
                    bioc_install_release = release, installed_during_setup = install_needed,
                    elapsed_seconds = proc.time()[["elapsed"]] - started)
write.table(setup, file.path(receipt, "setup.tsv"), quote = FALSE, row.names = FALSE, sep = "\t")
message("09F SpotClean install/load and synthetic spatial fit passed in ",
        round(setup$elapsed_seconds, 2), " seconds. Receipt: ", receipt)
