library(Seurat)
library(ggplot2)


xen <- readRDS("/project/simmons_hts/aantanav/_r_projects/xenium/_projects/1_CD_FISTULA/2_FISTULA_MERGE/merged.annotation.updated.md.disk.with.hcs.md.fixed.final.RDS")
dir <- "/project/simmons_hts/aantanav/_r_projects/xenium/_projects/1_CD_FISTULA/15_adjacency_dependent_gene_expression"

setwd(dir)

res <- list()

for ( file in list.files(dir, pattern="RDS_EPI")){
  
  res[[file]] <- readRDS(file)
  
}

mat <- t(do.call(rbind, res))

colnames(mat) <- stringr::str_split(colnames(mat), pattern = "RDS\\.", simplify=T)[, 2]
table(colnames(mat) %in% Cells(xen))

seurat <- CreateSeuratObject(mat[, Cells(xen)], meta.data=xen@meta.data)
rm(mat)
rm(res)
rm(xen)
gc()

seurat <- NormalizeData(seurat)


Idents(seurat) <- "CellAnnotation.Level0"

seurat.filt <- seurat[, seurat$nFeature_RNA > 50]
rm(seurat)
gc()
seurat.filt <- ScaleData(seurat.filt, features=rownames(seurat.filt))
saveRDS(seurat.filt, file=file.path(dir, "xenium_filt_with_epi_adjacency.RDS"))



#TODO - specifically look at Stromal subsets

Idents(seurat.filt) <- "CellAnnotation.Level3.Corrected"
mk <- FindAllMarkers(seurat.filt, max.cells.per.ident = 1000)

seurat.filt <- RenameIdents(seurat.filt, "FAS Cycling" = "FAS-CC", 
                            "Stromal 3/FAS" = "FAS-FOZ", "FAS MMP1/3+ IL11+" = "FAS-LAZ",
                            "FAS MMP1/3+" ="FAS-ALC", "Stromal 4"="FAS-LOC"
)



genes <- unique(c("ASCL2", "MKI67", "ATOH1", "LGR5", "OLFM4", "REG1A", "REG1B",
                  "NOS2", "PI3", "DUOX2", "FABP1",  "CHGA", "CHGB",
                  "DMBT1", "LEFTY1", "LCN2", "MUC6", "MUC5B", "SMOC2", "IDO1",
                  "AREG", "DUOXA2", "MUC2", "ATOH1", "STMN1", "MKI67", "SPIB",
                   "BEST4",  "SHH", "MUC12", "MUC17", "MUC2" ,"MUC3A" ,
                  "MUC5B" ,  "IHH",   "ISG15"  , "ISG20", "GP2" ,"GUCA2A" ,  "GUCA2B",
                  "CLDN3" , "CLDN4",  "CLDN15", "BEST4", "SPDEF", "SPINK1",  "CEACAM1", "TNFRSF11B"))


plot <- Seurat::DotPlot(seurat.filt, features=genes, idents=c("Stromal 2", "FAS-LOC",
                                                              "FAS-CC",
                                                              "FAS-FOZ", 
                                                              "FAS-ALC",
                                                              "Stromal 1",
                                                              "Muscularis Mucosa",
                                                              "Stromal 1 HHIP+",
                                                              "FAS-LAZ" ), col.min=-4, col.max=4)

mat <- reshape2::acast(plot$data[, 3:5], id~features.plot, value.var="avg.exp.scaled")
row_order <- hclust(dist(mat))$order
col_order <- hclust(dist(t(mat)))$order
plot$data$features.plot <- factor(plot$data$features.plot, levels=levels(plot$data$features.plot)[col_order])
plot$data$id <- factor(plot$data$id, levels=levels(plot$data$id)[row_order])

plot$data$id_type <- ifelse(plot$data$id %in% c("Stromal 2", "Muscularis Mucosa","Stromal 1 HHIP+", "Stromal 1"), yes="Mucosal", "FAS")


ggplot(plot$data, aes(id, features.plot, fill=avg.exp.scaled)) + geom_point(shape=22, size=8
) + scale_fill_distiller(palette="Spectral") + theme_classic(base_size = 12) + labs(x="", y="",
                                                                                    fill="Average\nExpression", size="Percent\nExpressing")  + theme(axis.text.x = element_text(angle=90))  + facet_grid(.~id_type, scales="free", space="free")




