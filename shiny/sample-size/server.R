
source("calculator.R")
library(shiny)
library(ggplot2)

shinyServer(function(input, output) {
    
    sample_pred <- reactive({
        # make sure inputs are valid
        need(list(input$numricP >= 0,
             input$numericDmin > 0,
             input$numericP + input$numericDmin <= 100),
             message = "conversion rate and/or change not valid")
        # convert inputs from percent to decimal
        sample_size(input$numericP / 100,
                    input$numericDmin / 100,
                    input$sliderAlpha / 100,
                    1 - input$sliderPower / 100,
                    input$type)
    })
    
    output$pred_out <- renderText({
        # round up with comma separator
        format(ceiling(sample_pred()$size), big.mark = ',', scientific = FALSE)
    })
    
    output$delta_out <- renderText({
        # round up with comma separator
        format(100 * round(sample_pred()$x.crit.delta, digits = 4), big.mark = ',', scientific = FALSE)
    })
    
    output$main_plot <- renderPlot({
        # get sample size values
        pred <- sample_pred()
        p <- pred$p; dmin <- pred$dmin; alt <- pred$alt
        # note x-axis values in ggplot for alpha and beta should be same
        z.alpha <- pred$z.alpha; z.beta <- pred$z.beta
        n <- pred$size; se.null <- pred$se.null; se.alt <- pred$se.alt
        
        # ggplot values for null hypothesis
        null <- density(rnorm(n, p, se.null))
        df_null <- data.frame(x = null$x, y = null$y)
        # base plot
        base_plot <- ggplot() +
            geom_ribbon(data = df_null,
                        aes(x = x, ymax = y, fill = "no change"),
                        ymin = 0, alpha = 0.35)
        
        # remainder of plot by test type
        if (alt == "two") {
            # x-axis values for type I error
            x1.alpha <- p - se.null * z.alpha
            x2.alpha <- p + se.null * z.alpha
            # x-axis values for type II error
            x1.beta <- p - dmin + se.alt * z.beta
            x2.beta <- p + dmin - se.alt * z.beta
            # ggplot data for alternative hypothesis
            alt1 <- density(rnorm(n, p - dmin, se.alt))
            df_alt1 <- data.frame(x = alt1$x, y = alt1$y)
            alt2 <- density(rnorm(n, p + dmin, se.alt))
            df_alt2 <- data.frame(x = alt2$x, y = alt2$y)
            
            final_plot <- base_plot +
                geom_ribbon(data = df_null[df_null$x <= x1.alpha, ],
                            aes(x = x, ymax = y, fill = "error: type I"),
                            ymin = 0, alpha = 0.3) +
                geom_ribbon(data = df_null[df_null$x >= x2.alpha, ],
                            aes(x = x, ymax = y, fill = "error: type I"),
                            ymin = 0, alpha = 0.3) +
                geom_ribbon(data = df_alt1, aes(x = x, ymax = y, fill = "true change"),
                            ymin = 0, alpha = 0.15) +
                geom_ribbon(data = df_alt1[df_alt1$x >= x1.beta, ],
                            aes(x = x, ymax = y, fill = "error: type II"),
                            ymin = 0, alpha = 0.3) +
                geom_ribbon(data = df_alt2, aes(x = x, ymax = y, fill = "true change"),
                            ymin = 0, alpha = 0.15) +
                geom_ribbon(data = df_alt2[df_alt2$x <= x2.beta, ],
                            aes(x = x, ymax = y, fill = "error: type II"),
                            ymin = 0, alpha = 0.3) +
                geom_vline(xintercept = x1.alpha, linetype = "dashed") +
                geom_vline(xintercept = x2.alpha, linetype = "dashed")
        }
        else if (alt == "less") {
            x.alpha <- p - se.null * z.alpha
            x.beta <- p - dmin + se.alt * z.beta
            alt <- density(rnorm(n, p - dmin, se.alt))
            df_alt <- data.frame(x = alt$x, y = alt$y)

            final_plot <- base_plot +
                geom_ribbon(data = df_null[df_null$x <= x.alpha, ],
                            aes(x = x, ymax = y, fill = "error: type I"),
                            ymin = 0, alpha = 0.3) +
                geom_ribbon(data = df_alt,
                            aes(x = x, ymax = y, fill = "true change"),
                            ymin = 0, alpha = 0.15) +
                geom_ribbon(data = df_alt[df_alt$x >= x.beta, ],
                            aes(x = x, ymax = y, fill = "error: type II"),
                            ymin = 0, alpha = 0.3) +
                geom_vline(xintercept = x.alpha, linetype = "dashed")
        }
        else {
            x.alpha <- p + se.null * z.alpha
            x.beta <- p + dmin - se.alt * z.beta
            alt <- density(rnorm(n, p + dmin, se.alt))
            df_alt <- data.frame(x = alt$x, y = alt$y)
            
            final_plot <- base_plot +
                geom_ribbon(data = df_null[df_null$x >= x.alpha, ],
                            aes(x = x, ymax = y, fill = "error: type I"),
                            ymin = 0, alpha = 0.3) +
                geom_ribbon(data = df_alt,
                            aes(x = x, ymax = y, fill = "true change"),
                            ymin = 0, alpha = 0.15) +
                geom_ribbon(data = df_alt[df_alt$x <= x.beta, ],
                            aes(x = x, ymax = y, fill = "error: type II"),
                            ymin = 0, alpha = 0.3) +
                geom_vline(xintercept = x.alpha, linetype = "dashed")
        }
        
        # FIXME old syntax doesn't worker with newest version shiny
        # limit range of x-axis/conversion rate to [0, 1]
        # plot_range <- ggplot_build(final_plot)$layout$panel_ranges[[1]]$x.range
        # x_min <- plot_range[1]
        # x_max <- plot_range[2]
        # if (x_min < 0) {x_min <- 0}
        # if (x_max > 1) {x_max <- 1}
        
        final_plot +
            scale_fill_manual(name = "test results",
                              values = c("no change" = "black",
                                         "true change" = "green4",
                                          "error: type I" = "red",
                                          "error: type II" = "orange")) +
            # coord_cartesian(xlim = c(x_min, x_max)) +
            xlab("conversion rate") +
            ylab("probability density") +
            theme(axis.text.y = element_blank())
    })
})
