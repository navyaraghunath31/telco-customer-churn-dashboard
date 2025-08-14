# Telco Customer Churn Dashboard & Prediction

## Project Overview
This project is a professional Streamlit dashboard designed to help telecom companies visualize and predict customer churn. It leverages machine learning and interactive data visualizations to provide actionable insights for business decision-makers.

---

## Business Use Case
### Problem Statement
Customer churn is a major challenge for telecom companies. Losing customers impacts revenue, increases acquisition costs, and can signal deeper issues with service or customer satisfaction. Predicting which customers are likely to leave allows companies to take proactive retention measures.

### Solution
This dashboard enables:
- **Churn Prediction:** Use historical customer data to predict the likelihood of churn for individual customers.
- **Data Exploration:** Visualize churn rates, feature distributions, and key drivers of churn.
- **Actionable Insights:** Identify high-risk customers and the factors contributing to churn, enabling targeted retention strategies.

### Benefits
- **Reduce Revenue Loss:** Retain valuable customers before they leave.
- **Optimize Marketing:** Focus retention campaigns on high-risk segments.
- **Improve Service:** Understand what drives churn and address root causes.

---

## Features
- Interactive dashboard with churn insights
- Predict customer churn using a Random Forest model
- Data visualizations: churn distribution, feature importance, correlation heatmap
- Modular codebase for easy maintenance
- Professional UI and branding

---

## How It Works
1. **Data Loading:** The app loads customer data from a CSV file.
2. **Preprocessing:** Categorical features are encoded and numeric features are scaled.
3. **Model Training:** A Random Forest classifier is trained to predict churn.
4. **Dashboard Tab:** Users can explore churn rates, feature distributions, and correlations.
5. **Predict Churn Tab:** Users input customer details to get a churn prediction and see how their scenario compares to the population.

---

## Project Structure
- `scripts/app.py`: Main Streamlit app
- `scripts/data_loader.py`: Data loading and cleaning
- `scripts/preprocessing.py`: Data preprocessing and encoding
- `scripts/model.py`: Model training and prediction
- `scripts/dashboard.py`: Dashboard tab logic and visualizations
- `scripts/predict.py`: Prediction tab logic
- `data/`: Raw data files
- `requirements.txt`: Python dependencies
- `README.md`: Quick start and overview
- `PROJECT_DOCUMENTATION.md`: Full documentation and business use case

---

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run scripts/app.py`
3. Open the dashboard in your browser and explore!

---

## Author & Contact
**Project by Navya R.**
Contact: navyaraghunath31@gmail.com

---

## Example Business Scenario
A telecom company wants to reduce churn. By using this dashboard, they:
- Upload their customer data
- Explore which features (e.g., contract type, monthly charges) are most associated with churn
- Predict churn for new or existing customers
- Use insights to design targeted retention offers for high-risk customers

---

## Customization & Extension
- Replace the dataset with your own customer data
- Retrain the model with new features or algorithms
- Add more visualizations or business metrics as needed

---

## License
Navya R
