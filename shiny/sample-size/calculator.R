
## This function calculates the minimum sample size
## required for an A/B test.
##
## The arguments to the function are:
## p = null hypothesis mean (also referred to as 'baseline conversion rate')
## dmin = difference of the alternative hypothesis mean ('minimum detectable effect')
## alpha = type I error rate
## beta = type II error rate
## alt = direction of alternative hypothesis
##
## This calculator also assumes a 2-sided test (variance is also maximized)

sample_size <- function(p, dmin, alpha, beta, alt = "two") {
    
    max_var <- FALSE
    
    # z-score (+) for type I error from alternative hypothesis
    if (alt %in% c("less", "greater")) {
        z.alpha <- qnorm(1 - alpha)
    }
    else {
        z.alpha <- qnorm(1 - alpha/2)
        max_var <- TRUE
    }
    # z-score for type II error
    z.beta <- qnorm(1 - beta)
    
    # alternative hypothesis mean
    if (max_var) {
        # if max_var (ie 2-sided) choose value closest to 0.5
        # this maximizes variance for larger sample size
        direction <- p - 0.5
        if (direction > 0) {p.a <- p - dmin}
        else {p.a <- p + dmin}
    }
    else {
        if (alt == "less") {p.a <- p - dmin}
        else {p.a <- p + dmin}
    }
    
    # calculate sample size
    se.null.numerator <- sqrt(2*p*(1 - p))
    se.alt.numerator <- sqrt(p*(1 - p) + (p.a)*(1 - (p.a)))
    n <- ((z.alpha*se.null.numerator + z.beta*se.alt.numerator) / dmin)^2
    
    # standard error for null and alternative hypothesis
    se.null <- se.null.numerator / sqrt(n)
    se.alt <- se.alt.numerator / sqrt(n)
    
    # calculate both critical values but show only one depending upon test type
    x.crit.delta <- z.alpha * se.null
    
    return(list("p" = p, "dmin" = dmin, "z.alpha" = z.alpha, "z.beta" = z.beta,
                "alt" = alt, "size" = n, "se.null" = se.null, "se.alt" = se.alt,
                "x.crit.delta" = x.crit.delta))
}


#alpha = 0.05; beta = 0.2; p = .53; dmin = .01
#output <- sample_size(p, dmin, alpha, beta)

# pwr.2p.test
#library(pwr)
#pwr.2p.test(h=ES.h(0.53,0.54),power=0.80,sig.level=0.05)
#pwr.2p.test(h=ES.h(0.53,0.52),power=0.80,sig.level=0.05)