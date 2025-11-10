Global Temperature Trend Analysis & Prediction Using Python 🌍📈

This project analyzes NASA GISTEMP climate data (1880–2024) to visualize global temperature trends and predict future anomalies using Python. It demonstrates data analysis, visualization, and simple predictive modeling.

Features & Functionalities

Data Collection & Cleaning

Used Pandas to read and clean raw CSV data from NASA GISTEMP datasets.

Handled missing data and formatted columns for analysis.

Data Analysis

Calculated annual and seasonal averages.

Explored temperature anomalies over time.

Used NumPy for numerical operations and calculations.

Visualization

Used Matplotlib to create line plots and trend graphs.

Highlighted global warming trends with clear, informative plots.

Prediction Model

Implemented linear regression using scikit-learn to forecast future temperature anomalies.

Learned about model limitations and the need for more advanced methods (e.g., polynomial regression).

Key Insights

The Earth shows a steady warming trend over the last century.

Simple linear models underestimate accelerated warming in recent decades.

Libraries Used

Pandas
 – for data manipulation and cleaning.

NumPy
 – for numerical calculations.

Matplotlib
 – for data visualization.

scikit-learn
 – for linear regression modeling.

How to Run

Clone the repository:

git clone https://github.com/Ahmed72007/globaltemp_analysis.git
cd globaltemp_analysis


Install required libraries (if not already installed):

pip install pandas numpy matplotlib scikit-learn


Run the Python script:

python global_temp_analysis.py


Check generated plots and predicted temperature trends.

Next Steps

Improve predictions using polynomial regression or weighted recent-year trends.

Add interactive visualizations using Plotly or Seaborn.

Compare predictions with IPCC climate models for validation.

Project Goals

Learn real-world data analysis with Python.

Practice data cleaning, visualization, and modeling.

Understand global warming trends through data.
