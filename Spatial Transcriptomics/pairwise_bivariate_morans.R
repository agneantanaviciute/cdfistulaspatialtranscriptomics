library(Seurat, lib="/ceph/project/simmons_hts/aantanav/bin/RLIBS_SEURAT5" )
library(BPCells, lib="/ceph/project/simmons_hts/aantanav/bin/RLIBS_SEURAT5" )
library(ggplot2)
library(sf)
library(spdep)
library(parallel)
options(future.globals.maxSize = 1e11)

source("/ceph/project/simmons_hts/aantanav/_r_projects/xenium/0_general/autocorrelation_functions.R")

setwd("/ceph/project/simmons_hts/aantanav/_r_projects/xenium/_projects/1_CD_FISTULA/7_autocorrelation")

merged <- readRDS("/ceph/project/simmons_hts/aantanav/_r_projects/xenium/_projects/1_CD_FISTULA/2_FISTULA_MERGE/merged.annotation.updated.md.disk.with.hcs.md.fixed.final.RDS")

#niche <- readRDS("/ceph/project/simmons_hts/aantanav/_r_projects/xenium/_projects/2_POLYPS/3_Niche_Analysis/merged.niches.exprs.RDS")
merged <- JoinLayers(merged)
merged$Image <- NA

for(image in Images(merged)){
  
  merged@meta.data[merged@images[[image]]@boundaries$centroids@cells, "Image"] <- image
}

args <- commandArgs(trailingOnly = TRUE)

ncores <- 10

img <- args[1]
print(img)

cells <- Cells(merged)[merged$Image == img ]

coords <- merged@images[[img]]@boundaries$centroids@coords[merged@images[[img]]@boundaries$centroids@cells %in% cells, ]
data <- merged@assays$XENIUM$data[, cells]

results <- CalculateMoranBivariateI(coords = coords, data = data)

saveRDS(results, file=paste0(img, "_morans.bivariate.matrices.per.image.pairwise.genes.RDS"))

