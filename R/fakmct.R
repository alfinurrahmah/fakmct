#' @title Fuzzy Adaptive Resonance Theory (ART) K-Means Clustering Technique
#'
#' @description Clustering data observation with hybrid method Fuzzy ART and K-Means
#'
#' @param input The input (vector) data observation. Should be numeric type of data.
#' @param rho Vigilance parameter in (0,1)
#' @param alpha Choice parameter alpha > 0
#' @param beta Learning rate in (0,1)
#' @param w_init Initial weight
#' @param max_epochs Maximum number of iterations
#' @param max_clusters Maximum number of clusters that allowed
#' @param eps Tolerance with default is 10^-6
#'
#' @return
#' \item{labels}{clusters label of each observations}
#' \item{size}{the size of each clusters that have been formed}
#' \item{clusters}{a list of observations in each clusters}
#' \item{centroids}{cluster centroids that are calculated by the mean value of the objects in each clusters}
#' \item{weights}{the model weight}
#' \item{params}{parameters that have been saved in the function}
#' \item{num_clusters}{number of cluster that have been formed}
#' \item{running.time}{time for running function}
#'
#'
#' @export
#' @importFrom stats runif
#'
#'@examples
#' library(fakmct)
#' # Using dataset iris
#' ## load data
#' data.inputs = iris[,-5]
#' true.labels = as.numeric(unlist(iris$Species))
#'
#' ## run model data
#' ex.iris<-fakmct(data.inputs, alpha = 0.3, rho = 0.5, beta = 1, max_epochs = 50, max_clusters = 5)
#' ex.iris$labels
#' ex.iris$size
#' ex.iris$centroids
#' ex.iris$params
#'
#' ## plot data
#' plot(data.inputs, col = ex.iris$labels, pch = true.labels,
#'      main = paste0("Dataset: Iris"))
#'
#' # Using data IPM 2019
#'
#' ## load simulate data IPM
#' data("simulatedataIPM")
#' dt <- simulatedataIPM
#'
#' ## run model data IPM
#' mod.fakm<-fakmct(dt, alpha = 0.3, rho = 0.5, beta = 0.1, max_epochs = 50, max_clusters = 5)
#' mod.fakm$labels
#' mod.fakm$size
#' mod.fakm$centroids
#' mod.fakm$params
#'
#' ## plot data IPM
#' plot(dt, col = mod.fakm$labels, pch=mod.fakm$labels, main = paste0("Dataset IPM"))




fakmct <- function(input, rho, alpha, beta, w_init = NA,
                   max_epochs = 1000, max_clusters = 1000, eps = 10^-6)
{
  start_time <- Sys.time()
  ## save parameters
  params = list()
  params[["rho"]]         = rho
  params[["alpha"]]       = alpha
  params[["beta"]]        = beta
  params[["max_epochs"]]  = max_epochs
  params[["w_init"]]      = w_init


  ## normalize data
  normalize_data <- function(input)
  {
    normalized <- apply(X = input,MARGIN = 2,FUN = function(col){(col-min(col))/(max(col)-min(col))})
    return(normalized)
  }

  ## complement coded conversion
  complement_coded_conv <- function(normalized_data)
  {
    dataset <- cbind(normalized_data,1-normalized_data)
    return(dataset)
  }

  ## the process of selecting winning category with the activation function and checking the vigilance parameters
  category_winner <- function(input, w, alpha, rho)
  {
    #w
    m = dim(w)[1] # number of clusters
    matches = rep(0,m)
    # matches = apply(X = w,MARGIN = 1, FUN = function(weight){return(choice_function(input, weight, alpha))})
    for (j in 1:m) {
      matches[j] = choice_function(input, w[j,], alpha)
    }

    match_iterations = 0
    while (match_iterations<length(matches))
    {
      # select the winning category uses which.max
      winner = which.max(matches) #location of the first max value
      if(vigilance_check(input,w[winner,],rho))
      {
        # induce resonance
        return(winner)
      } else
      {
        # failed to match, try another Tj
        matches[winner] <- 0 # RESET
        match_iterations <- match_iterations+1
      }
    }
    return(length(matches)+1)
  }

  ## learning pattern and creating new cluster
  learn_pattern <- function(input, w, rho, alpha, beta, num_clusters, max_clusters,
                            output.learn = list(w = NA, num_clusters = NA, winner = NA))
  {
    # found the location of winning category
    winner <- category_winner(input = input, w = w, alpha = alpha, rho = rho)

    if(winner > num_clusters)
    {
      # create a new category
      num_clusters <- num_clusters + 1

      # was it reached the maximum cluster count?
      if(num_clusters > max_clusters) stop("Error. The maximum cluster count has been reached!")

      w = rbind(w,input)
    }

    # update the old weight
    w[winner,] <- update_weight(input, w[winner,], beta)

    output.learn$winner = winner
    output.learn$w = w
    output.learn$num_clusters = num_clusters
    return(output.learn)
  }

  ## recalculating centroids by the mean value of the objects in each cluster
  recalculate_centroids <- function(centroids, clusters, k)
  {
    for (i in 1:k)
    {
      centroids[i,] = apply(X=clusters[[i]], MARGIN=2, FUN = mean)
    }
    return(centroids)
  }

  ## assiging each observations to nearest cluster with euclidean distance
  assigning_cluster <- function(input, centroids, k)
  {
    m = dim(input)[1]
    cluster_labels = matrix(rep(NA, m))
    nearest = NA
    for (j in 1:m)
    {
      d = matrix(0,nrow=k)
      for (i in 1:k)
      {
        d[i] <- linalg_norm(input[j,],centroids[i,])
      }
      nearest = which.min(d)
      cluster_labels[j] = nearest
    }
    return(cluster_labels)
  }


  ## Training Fuzzy ART neural network
  fuzzyart_training <- function(input, rho, alpha, beta, w_init = NA,  max_epochs = 1000, max_clusters = 20, eps = 10^-6)
  {

    start_time = Sys.time()
    ## save parameters
    params = list()
    params[["rho"]]         = rho
    params[["alpha"]]       = alpha
    params[["beta"]]        = beta
    params[["max_epochs"]]  = max_epochs
    params[["w_init"]]      = w_init

    ## check input data

    if(is.data.frame(input)){(input=as.matrix(input))}
    if (is.numeric(input))
    {
      # normalize data input
      normalized_data <- normalize_data(input)
    } else {stop("Error. Input data is not numeric")}


    # complement code conversion
    dataset <- complement_coded_conv(normalized_data)

    # determine the size of features/variables data
    num_features <- ncol(input)

    # check parameter
    if(rho < 0 && rho > 1) stop("Error. Rho must be between 0 and 1")
    if(beta < 0 && beta > 1) stop("Error. Beta must be between 0 and 1")
    if(alpha<=0) stop("Error. Alpha must be greater than 1")

    # init weight
    #w_init = NA
    if(is.na(w_init))
    {
      w <- matrix(rep(1,num_features*2),nrow = 1)
    } else if (dim(w_init)[2]==num_features*2)
    {
      w <- w_init
    } else {stop("Error. Initial weight dimension is not the same as 2*num of features")}

    num_clusters <- dim(w)[1]

    #init variables
    max_clusters = max_clusters
    num_clusters_old = num_clusters
    cluster_labels = matrix(rep(NA, dim(dataset)[1]))
    iterations = 0
    w_old = w+1
    train_output_fa = list(w = NA, num_clusters = NA, winner = NA)
    index = 1:dim(dataset)[1]

    while (((sum(w_old-w)^2)>eps^2) & (iterations < max_epochs))
    {

      w_old = w

      for(i in index)
      {
        train_output_fa <- learn_pattern(dataset[i,], w, rho, alpha, beta, num_clusters,
                                         max_clusters, output.learn =  train_output_fa)

        cluster_labels[i] <- train_output_fa$winner
        dimnames(cluster_labels)[1]<-dimnames(input)[1]
        w <- train_output_fa$w
        num_clusters <- train_output_fa$num_clusters

      }
      iterations = iterations+1

      if(num_clusters_old != num_clusters)
      {
        w_old = rbind(w_old,replicate(n = num_features*2, rep(1,num_clusters-num_clusters_old)))
        num_clusters_old = num_clusters
      }
    }

    fa_clusters = list(matrix(NA,nrow=num_clusters, ncol=num_features))
    size = matrix(NA,num_clusters)
    cent= matrix(NA,nrow=num_clusters,ncol=ncol(input))
    for(k in 1:num_clusters)
    {
      fa_clusters[[k]] <- matrix(input[which(cluster_labels == k),], ncol = num_features)
      dimnames(fa_clusters[[k]])[1]<-dimnames(input[which(cluster_labels == k),])[1]
      dimnames(fa_clusters[[k]])[2]<-dimnames(input)[2]
      if(is.null(nrow(fa_clusters[[k]])))
      {
        size[k] <- 0
      } else {size[k]<-nrow(fa_clusters[[k]])}
    }

    fa_centroids = matrix(NA,nrow=num_clusters,ncol=ncol(input))
    fa_centroids <- recalculate_centroids(fa_centroids,fa_clusters,num_clusters)
    dimnames(fa_centroids)[2]<-dimnames(input)[2]

    end_time <- Sys.time()
    running_time = end_time-start_time

    FuzzyArt_Out = list()
    FuzzyArt_Out[["labels"]] = cluster_labels
    FuzzyArt_Out[["size"]] = size
    FuzzyArt_Out[["clusters"]] = fa_clusters
    FuzzyArt_Out[["centroids"]] = fa_centroids
    FuzzyArt_Out[["weights"]] = w_old
    FuzzyArt_Out[["params"]] = params
    FuzzyArt_Out[["iterations"]] = iterations
    FuzzyArt_Out[["num_clusters"]] = num_clusters
    FuzzyArt_Out[["running_time"]]= running_time
    return(FuzzyArt_Out)
  }

  ## Fuzzy ART Testing

  fuzzyart_testing <- function(input, FuzzyArt_Out)
  {
    start_time = Sys.time()
    if(is.null(FuzzyArt_Out)) stop("Error. Need to specify FuzzyArt_Out as a FuzzyART model as trained by FuzzyART_train")

    # check parameter
    if(is.data.frame(input)){(input=as.matrix(input))}
    if (is.numeric(input))
    {
      # normalize data input
      normalized_data <- normalize_data(input)
    } else {stop("Error. Input data is not numeric")}


    # complement code conversion
    dataset <- complement_coded_conv(normalized_data)
    num_clusters = FuzzyArt_Out$num_clusters

    cluster_labels = matrix(rep(NA, dim(input)[1]))
    cluster_labels = matrix(apply(dataset, MARGIN = 1,FUN = function(dataset){return(category_winner(input = dataset,w = FuzzyArt_Out$w,
                                                                                                     alpha = FuzzyArt_Out$params$alpha,
                                                                                                     rho = FuzzyArt_Out$params$rho))}))
    num_clusters = FuzzyArt_Out$num_clusters
    dimnames(cluster_labels)[1]<-dimnames(input)[1]

    fa_clusters = list(matrix(NA,nrow=num_clusters, ncol=ncol(input)))
    size = matrix(NA,num_clusters)
    cent= matrix(NA,nrow=num_clusters,ncol=ncol(input))
    for(k in 1:num_clusters)
    {
      fa_clusters[[k]] <- matrix(input[which(cluster_labels == k),], ncol = ncol(input))
      dimnames(fa_clusters[[k]])[1]<-dimnames(input[which(cluster_labels == k),])[1]
      dimnames(fa_clusters[[k]])[2]<-dimnames(input)[2]
      if(is.null(nrow(fa_clusters[[k]])))
      {
        size[k] <- 0
      } else {size[k]<-nrow(fa_clusters[[k]])}
    }

    fa_centroids = matrix(NA,nrow=num_clusters,ncol=ncol(input))
    fa_centroids <- recalculate_centroids(fa_centroids,fa_clusters,num_clusters)
    dimnames(fa_centroids)[2]<-dimnames(input)[2]

    end_time <- Sys.time()
    running_time = end_time-start_time

    FuzzyArt_Res = list()
    FuzzyArt_Res[["labels"]] = cluster_labels
    FuzzyArt_Res[["size"]] = size
    FuzzyArt_Res[["clusters"]] = fa_clusters
    FuzzyArt_Res[["centroids"]] = fa_centroids
    FuzzyArt_Res[["weights"]] = FuzzyArt_Out$w
    FuzzyArt_Res[["params"]] = FuzzyArt_Out$params
    FuzzyArt_Res[["num_clusters"]] = num_clusters
    FuzzyArt_Res[["running_time"]]= running_time
    return(FuzzyArt_Res)
  }

  ## K-Means Clustering
  kmeans_train <- function(input, k, centroids = NULL, max_iter = 1000)
  {
    start_time = Sys.time()
    iterations = 0
    clust= list(matrix(NA,nrow=k, ncol=dim(input)[2]))
    labels=matrix(rep(NA, dim(input)[1]))
    size = matrix(NA,k)
    if(is.data.frame(input)){(input=as.matrix(input))}

    if(is.null(centroids))
    {
      centroids = matrix(NA,nrow=k,ncol=ncol(input))
      for (i in 1:k) {
        rand <- round(runif(1,1,nrow(input)),0)
        centroids[i,] =  rbind(input[rand,])
      }
    } else {centroids=centroids}

    while(iterations <= max_iter)
    {
      centroids_old = centroids
      labels = assigning_cluster(input, centroids, k)
      for(i in 1:k)
      {
        clust[[i]] <- matrix(input[which(labels == i),], ncol = ncol(input))
        dimnames(clust[[i]])[1]<-dimnames(input[which(labels == i),])[1]
        dimnames(clust[[i]])[2]<-dimnames(input)[2]
        if(is.null(nrow(clust[[i]])))
        {
          size[i] <- 0
        } else {size[i]<-nrow(clust[[i]])}
      }
      centroids = recalculate_centroids(centroids_old, clust, k)
      if(all(centroids_old == centroids, na.rm = T)==TRUE)
      {
        break
      } else
      {
        iterations = iterations+1
      }
    }
    dimnames(labels)[1]<-dimnames(input)[1]


    end_time = Sys.time()
    running_time = end_time-start_time
    Kmeans_Res = list()
    Kmeans_Res$labels = labels
    Kmeans_Res$size = size
    Kmeans_Res$clusters = clust
    Kmeans_Res$centroids = centroids
    Kmeans_Res$running_time = running_time
    return(Kmeans_Res)
  }


  FuzzyArt_Out = list(NA)
  FuzzyArt_Out <- fuzzyart_training(input, rho, alpha, beta, w_init, max_epochs, max_clusters)
  FuzzyArt_Res = list(NA)
  FuzzyArt_Res <- fuzzyart_testing(input,FuzzyArt_Out)
  Kmeans_Res = list (NA)
  Kmeans_Res <- kmeans_train(input, k = FuzzyArt_Res$num_clusters, centroids = FuzzyArt_Res$centroids)

  end_time <- Sys.time()
  running_time = end_time - start_time

  fakmct_res = list()
  fakmct_res[["labels"]]=Kmeans_Res$labels
  fakmct_res[["size"]]=Kmeans_Res$size
  fakmct_res[["clusters"]]=Kmeans_Res$clusters
  fakmct_res[["centroids"]]=Kmeans_Res$centroids
  fakmct_res[["weights"]]= FuzzyArt_Res$w
  fakmct_res[["params"]] = params
  fakmct_res[["num_clusters"]] = FuzzyArt_Res$num_clusters
  fakmct_res[["running_time"]]= running_time
  return(fakmct_res)

}



