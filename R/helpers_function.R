
#' Fuzzy And Function
#'
#' @param inputA First input vector
#' @param inputB Second input vector. Must be of the same dimension as inputA.
#'
#' @return Returns the Fuzzy AND of two input values in a vector.
#' @export
#'
#' @examples fuzzy_and(0, -1) # = -1
#' fuzzy_and(0, 1) # = 0
#' fuzzy_and(1, 2) # = 1
#' fuzzy_and(1, 1) # = 1
#' fuzzy_and(c(0.5, 0.75), c(1.5, 1)) # = c(0.5,0.75)

fuzzy_and <- function(inputA, inputB)
{
  comb <- cbind(inputA,inputB)
  return(apply(X = comb,FUN = min,MARGIN = 1))
}

#' Fuzzy Norm
#'
#' @param input The input (vector) data observation
#'
#' @return Returns the Fuzzy norm results of input values
#' @export
#'
#' @examples a = c(-1,-3,4,5)
#' fuzzy_norm(a) # = 13

fuzzy_norm <- function(input)
{
  return(sum(abs(input)))
}

#' Choice Function
#'
#' @description Calculates the similarity between the input pattern I and all of saved categories.
#'
#' @param input The input (vector) data observation
#' @param category_w The current category weight
#' @param alpha Choice parameter alpha > 0
#'
#' @return Returns the vector of Tj choice activation function
#' @export


choice_function <- function(input, category_w, alpha)
{
  Tj <- fuzzy_norm(fuzzy_and(input, category_w))/(alpha+fuzzy_norm(category_w))
  return(Tj)
}

#' Match function
#'
#' @param input The input (vector) data observation
#' @param category_w The current category weight
#'
#' @return Returns the vector of match Sj that will be used to check the vigilance parameter
#' @export

match_function <- function(input, category_w)
{
  Sj <- fuzzy_norm(fuzzy_and(input,category_w))/fuzzy_norm(input)
  return(Sj)
}

#' Vigilance check
#'
#' @param input The input (vector) data observation
#' @param category_w The current category weight
#' @param rho Vigilance parameter (0,1)
#'
#' @return Returns Boolean value (True or False) as a result of checking the match Sj vector passed the vigilance parameter or not
#' @export

vigilance_check <- function(input, category_w, rho)
{
  vcheck <- match_function(input,category_w) >= rho
  return(vcheck)
}

#' Update weight
#'
#' @param input The input (vector) data observation
#' @param category_w The current category weight
#' @param beta Learning rate in (0,1)
#'
#' @return Returns the updated weight
#' @export

update_weight <- function(input, category_w, beta)
{
  w_new <- beta*fuzzy_and(input,category_w) + (1-beta)*category_w
  return(w_new)
}

#' Linear Algebra for Euclidean distance
#' @param inputA First input vector
#' @param inputB Second input vector. Must be of the same dimension as inputA.
#'
#' @return Returns the calculation results by squares of distances between two input values
#'
#' @export
#' @examples
#' a <- c(-3,-2,-1,3,3,2,3)
#' b <- c(-3,-2,-1,0,1,2,3)
#' linalg_norm(a,b) # = 3.605

linalg_norm <- function(inputA,inputB)
{
  return(sqrt(sum(abs(inputA-inputB)^2)))
}


