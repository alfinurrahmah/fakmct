% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/helpers_function.R
\name{choice_function}
\alias{choice_function}
\title{Choice Function}
\usage{
choice_function(input, category_w, alpha)
}
\arguments{
\item{input}{The input (vector) data observation}

\item{category_w}{The current category weight}

\item{alpha}{Choice parameter alpha > 0}
}
\value{
Returns the vector of Tj choice activation function
}
\description{
Calculates the similarity between the input pattern I and all of saved categories.
}
