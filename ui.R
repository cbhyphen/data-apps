
library(shiny)
library(shinythemes)

shinyUI(fluidPage(
    
    theme = shinytheme("united"),
    # title and text
    fluidRow(align = "center",
        titlePanel("Sample Size for your A/B Test"),
        column(10, offset = 1,
               p("This shiny app calculates the sample size for your A/B test.
          It also displays the change in rate at which to reject the null
          hypothesis.  The variable underlying the rate is assumed to follow
          a binomial distribution.
          There is also a plot to show probability densities of the hypotheses 
          along with errors.  Use the controller to select the parameters for
          your test and hit the", code("submit"), "button to calculate the
          sample size.  All values in the controller are as a percent. 
          (note that if the sample size is very
          large, it may take ggplot a moment to render)")
               )
    ),
    # horizontal ruler
    tags$hr(),
    # selector/controller for sample size and plot
    fluidRow(
        column(4,
               numericInput("numericP", "current baseline conversion rate (%)",
                            value = 15, min = 1, max = 98, step = 0.5),
               numericInput("numericDmin", "minimum detectable change (%)",
                            value = 2, min = 0.5, max = 20, step = 0.5)
               ),
        column(4,
               sliderInput("sliderAlpha", "significance level (%)",
                           value = 5, min = 1, max = 20, step = 0.5),
               sliderInput("sliderPower", "statistical power (%)",
                           value = 80, min = 50, max = 98, step = 0.5)
               ),
        column(4,
               radioButtons("type", "Test for change in:",
                            c("either direction" = "two",
                              "one direction - less" = "less",
                              "one direction - greater" = "greater")),
               submitButton("submit")
               )
    ),
    tags$hr(),
    # sample size output
    # fluidRow(align = "center",
    #          h4("sample size:"),
    #          h1(textOutput("pred_out")),
    #          h4("critical delta:"),
    #          h1(textOutput("delta_out"))
    # ),
    # horizontal ruler
    # tags$hr(),
    # critical delta
    fluidRow(
        column(6,
               align = "center",
               h4("sample size:"),
               h1(textOutput("pred_out"))
        ),
        column(6,
               align = "center",
               h4("critical delta (%):"),
               h1(textOutput("delta_out"))
        )
    ),
    # horizontal ruler
    tags$hr(),
    # density plot
    plotOutput("main_plot")
))
