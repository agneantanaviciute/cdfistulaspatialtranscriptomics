library(Seurat, lib="/ceph/project/simmons_hts/aantanav/bin/RLIBS_SEURAT5" )
library(BPCells, lib="/ceph/project/simmons_hts/aantanav/bin/RLIBS_SEURAT5" )
library(ggplot2)
library(sf)
library(spdep)
library(parallel)
options(future.globals.maxSize = 1e11)


CalculateMoranIMatrix <- function(coords, data){
  
  # Create a k-nearest neighbors spatial weights matrix
  nb <- knn2nb(knearneigh(coords, k = 4))
  listw <- nb2listw(nb, style = "W", zero.policy = TRUE)
  
  results <- sapply(X = 1:nrow(x = data), FUN = function(x) {
    print(x)
    tryCatch(expr =  moran.test(scale(data[x, ]), listw )$estimate, error = function(x) c(1, 
                                                                              1, 1))
  })
  
  colnames(results) <- rownames(data)
  results
  
}

CalculateMoranBivariateI <- function(coords, data){
  
  nb <- knn2nb(knearneigh(coords, k = 4))
  listw <- nb2listw(nb, style = "W", zero.policy = TRUE)
  
  res <- matrix(nrow=nrow(data), ncol=nrow(data), dimnames = list(rownames(data), rownames(data)))
  
  
  for( i in 1:nrow(data)){
    
    print(i)
    
    for( j in 1:nrow(data)){
      tryCatch({
        z_var1 <- scale(data[i, ])
        z_var2 <- scale(data[j, ])
        lag_z_var2 <- lag.listw(listw, z_var2)
        I <- sum(z_var1 * lag_z_var2) / sum(z_var1^2)
        res[i, j] <- I
      }, error=function(x){})

    }
  }
  
  return(res)
    
}


CalculateMoranBivariateIParallel <- function(coords, data, ncores){
  
  cl <- makeCluster(ncores)
  
  nb <- knn2nb(knearneigh(coords, k = 4))
  listw <- nb2listw(nb, style = "W", zero.policy = TRUE)
  
  compute <- function(i, data, listw){
    
    results <- numeric(nrow(data))
    
    for( j in 1:nrow(data)){
      tryCatch({
        z_var1 <- scale(data[i, ])
        z_var2 <- scale(data[j, ])
        lag_z_var2 <- lag.listw(listw, z_var2)
        I <- sum(z_var1 * lag_z_var2) / sum(z_var1^2)
        results[j] <- I
      }, error=function(e){
        results[j] <- NA
      })

    }
    
    return(results)
  }
  
  clusterExport(cl, varlist = c("compute", "listw", "data"))
  result <- parLapply(cl, 1:nrow(data), function(i) compute(i, data, listw))
  stopCluster(cl)
  result <- do.call(rbind, result)
  rownames(result) <- rownames(data)
  colnames(result) <- rownames(data)
  
  return(result)
  
}



