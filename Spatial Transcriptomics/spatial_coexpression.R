library(Seurat, lib="/ceph/project/simmons_hts/aantanav/bin/RLIBS_SEURAT5" )
library(pheatmap)
library(ggplot2)
library(Matrix,lib="/ceph/project/simmons_hts/aantanav/bin/RLIBS_SEURAT5")

#' Creates a product of all pairwise gene combinations and their co-expression
#' This is non-symmetrical- the product is between gene A in cell and gene B in surrounding cells
#'@param object seurat object
#'@param fov name of the image/fov to compute neighbours from
#'@param neighbors.k nearest neighbours to use
#'@param assay.counts which spatial assay to pull counts from
#'@return matrix of nearest neighbour aggregated counts for each feature, excluding the cell itself
#'
CreatePairwiseGeneProductMatrix.PerImage <- function(
    object,
    fov,
    neighbors.k = 30,
    assay.counts = "XENIUM"
) {
  # find neighbors based on tissue position
  coords <- GetTissueCoordinates(object[[fov]], which = "centroids")
  cells <- coords$cell
  rownames(coords) <- cells
  coords <- as.matrix(coords[ , c("x", "y")])
  neighbors <- FindNeighbors(coords, k.param = neighbors.k)
  neighbors$nn <- neighbors$nn[cells, cells]
  
  diag(neighbors$nn) <- 0 # dont count transcriptome of the cell itself, just neighbours?
  
  mt <- object[[assay.counts]]@counts[, cells]
  
  sum.mtx <- as.matrix(neighbors$nn %*% t(mt))
  
  sum.mtx <- t(sum.mtx)
  # Get the dimensions of the matrices
  nrow_A <- nrow(sum.mtx)
  ncol_A <- ncol(sum.mtx)
  nrow_B <- nrow(mt)
  ncol_B <- ncol(mt)
  
  # Initialize a matrix to store the results
  result_matrix <- matrix(0, nrow = nrow_A * nrow_B, ncol = ncol_A)
  
  # Compute the pairwise product of rows
  row_index <- 1
  rownames <- c()
  for (i in 1:nrow_A) {
    print(i)
    for (j in 1:nrow_B) {
      result_matrix[row_index, ] <- sum.mtx[i, ] * mt[j, ]
      row_index <- row_index + 1
      rownames[row_index] <- paste0(rownames(sum.mtx)[i], "-" ,  rownames(mt)[j])
    }
  }
  

  rownames(results_matrix) <- rownames

  return(result_matrix)
}



