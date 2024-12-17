library(Seurat)

source("/ceph/project/simmons_hts/aantanav/_r_projects/xenium/0_general/niche_functions.R")

args <- commandArgs(trailingOnly = TRUE)

print(args)
print(args[1])

xen <- readRDS(file.path("/project/simmons_hts/aantanav/_r_projects/xenium/_projects/1_CD_FISTULA/12_individual_sample_section_export/", args[1]))

#clusters_to_keep <- c("FAS Cycling", "FAS MMP1/3+", "FAS MMP1/3+ IL11+")
clusters_to_keep <- unique(xen$CellAnnotation.Level3.Corrected[xen$CellAnnotation.Level0 == "Epithelium"])

cells <- BuildCellSpecificNicheExpressionAssay(xen, group.by = "CellAnnotation.Level3.Corrected", 
                                              idents = clusters_to_keep, assay.counts = "XENIUM", neighbors.k = 20)

saveRDS(cells, file=file.path("/ceph/project/simmons_hts/aantanav/_r_projects/xenium/_projects/1_CD_FISTULA/15_adjacency_dependent_gene_expression/",
                              paste0(args[1], "_EPI_adjacency.RDS")))


