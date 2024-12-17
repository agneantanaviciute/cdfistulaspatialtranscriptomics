library(Seurat)
library(ggplot2)


xen <- readRDS("/project/simmons_hts/aantanav/_r_projects/xenium/_projects/1_CD_FISTULA/2_FISTULA_MERGE/merged.annotation.updated.md.disk.with.hcs.md.fixed.final.RDS")
dir <- "/project/simmons_hts/aantanav/_r_projects/xenium/_projects/1_CD_FISTULA/15_adjacency_dependent_gene_expression"

setwd(dir)

res <- list()

for ( file in list.files(dir, pattern="RDS_FAS")){
  
  res[[file]] <- readRDS(file)
  
}

mat <- t(do.call(rbind, res))

colnames(mat) <- stringr::str_split(colnames(mat), pattern = "RDS\\.", simplify=T)[, 2]
table(colnames(mat) %in% Cells(xen))


seurat <- CreateSeuratObject(mat[, Cells(xen)], meta.data=xen@meta.data)
seurat <- NormalizeData(seurat)


Idents(seurat) <- "CellAnnotation.Level0"

seurat.filt <- seurat[, seurat$nFeature_RNA > 50]
seurat.filt
mk <- FindAllMarkers(seurat.filt, max.cells.per.ident = 5000)


seurat.filt <- ScaleData(seurat.filt, features=rownames(seurat.filt))

genes <- unique(c("CXCL13",  "F3", "WNT5A", "NRG1", "MMP1", "MMP3", "PRRX1", "IL11",
           "RUNX2", "COL7A1", "CHI3L2", "TWIST1", "CXCL1", "CXCL2", "MMP19", "MMP9",
           "STAT1", "WNT2", "DLL1", "BMP1", "PDGFRA", "FZD1", "MME", "WNT2", "IL6", "TNFSF11", "WNT5B"))

plot <- Seurat::DotPlot(seurat.filt, features=genes)
mat <- reshape2::acast(plot$data[, 3:5], id~features.plot, value.var="avg.exp.scaled")
row_order <- hclust(dist(mat))$order
col_order <- hclust(dist(t(mat)))$order
plot$data$features.plot <- factor(plot$data$features.plot, levels=levels(plot$data$features.plot)[col_order])
plot$data$id <- factor(plot$data$id, levels=levels(plot$data$id)[row_order])


ggplot(plot$data, aes(id, features.plot, fill=avg.exp.scaled)) + geom_point(shape=22, size=8
) + scale_fill_distiller(palette="Spectral") + theme_light(base_size = 16) + labs(x="", y="",
                                                                                  fill="Average\nExpression", size="Percent\nExpressing")  + theme(axis.text.x = element_text(angle=90)) + coord_flip()





saveRDS(seurat.filt, file=file.path(dir, "xenium_filt_with_fas_adjacency.RDS"))


VlnPlot(seurat.filt, "NRG1", alpha = 0)
VlnPlot(seurat.filt, "MMP1", alpha = 0)
VlnPlot(seurat.filt, "MMP3", alpha = 0)
VlnPlot(seurat.filt, "IL11", alpha = 0)
VlnPlot(seurat.filt, "WNT5A", alpha = 0)
VlnPlot(seurat.filt, "F3", alpha = 0)




