# EFFECT OF INTERNATIONAL AID ON ECONOMIC GROWTH

The effectiveness of International aid on economic growth of developing countries has been studied by economists for several decades. The importance of the topic can be supported by its implications in poverty rate reduction in developing countries.

The main reason to choose the topic is the desire to contribute to make the world a better place for everyone. In this project, previous empirical studies on foreign aid and the economic growth have been analyzed and varying results found.

# TABLE OF CONTENTS

[Research objective](#research-objective)

[Methodology](#methodology)

[Tools](#tools)

[Data sources](#data-sources)

[Variable description](#variable-description)

## Research objective

1. Analyzing the effects of foreign aid to low-income countries.

2. Assessing the impact of support on low-income countries.

3. Evaluating the relationship linking foreign aid and economic growth of developing countries.

4. Estimating openness index in the model.

5. Correlating the country's economy by evaluating the influences for trade on domestic activities.

## Methodology

The technique used for estimation is paanel least squares. The reason to choose the following technique is based on background research and better results on panel data.

The model is derived from production function which includes domestic capital, labour input foreign aid. A general representation of the production function can be written as follows:

Y = f (L, K, A)
Y – Gross Domestic Product (GDP) in real terms
L – labor input
K – domestic capital stock
A – foreign aid
It is assumed to be linear in logs, taking logs and differencing the model. The growth rate of real GDP is determined by the following expression:
y=a+βl+ δk+ φa


## Tools

1. SQLite: Combining columns, removing inconsistencies, check for missing data.

2. Tableau: Creating Visualizations, mainly using the tool for basic descriptive analysis.

3. Python: using several packages and statistical tools for creating a model and performing regression.

## Data Sources

In order to estimate my model, a panel of aggregate data on worldwide flows of foreign aid to developing countries was acquired. The source of our data is the World Bank Development Indicators (WDI). The data set covers the period for over 1990 to 2017 which includes 83 countries (World Sample) for which foreign aid and other control variables are reported. 

## Variable description

1. Growth- The economic growth rate of the country is the dependent variables. The variable name in our data set is GDP growth and it was already in the required form.

2. Foreign aid- The foreign aid is our variable of interest, taken as a percentage of GDP.

3. Savings- Gross saavings is also used as a percentage of GDP in real terms.

4. Labor force- In the data set, the label is Population growth (annual %).

5. Wopen- The variable is used as a proxy for trade distortions and measures for trade policy. Wopens dominates other indexes by considering the significance of both the country's trade balance and intensity. To illustrate, Wopen index is calculated as a standard openness index (X+I)/GDP divided by current account balance (X-I)/GDP.
