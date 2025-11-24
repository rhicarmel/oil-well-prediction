# 🛢️ Oil Well Profitability Prediction

## Overview
A complete machine learning analysis identifying the **most profitable and least risky region** for future oil well development.  
Using geological data from three regions, the project predicts oil reserves, evaluates profit potential, and estimates investment risk.

**Goal:** Recommend the optimal region for drilling based on expected profit and loss probability.  
**Best Region:** **Region 1**  
- Highest average profit  
- Lowest RMSE  
- Lowest probability of financial loss  

### Run Notebook
[![Run Notebook](https://img.shields.io/badge/📓_Open_Notebook-orange?style=for-the-badge)](./OilWells.ipynb)

### Run the interactive app on Streamlit
[![Streamlit App](https://img.shields.io/badge/🚀_Open_Streamlit_App-ff4b4b?style=for-the-badge)](https://oil-well-prediction-rhi-222.streamlit.app/)
<!-- Update the URL above after you deploy the app -->

---

## Functionality
- Loads and cleans geological datasets from three regions  
- Trains a **Linear Regression** model to predict oil reserves  
- Measures model accuracy using **RMSE**  
- Simulates profit using the **top 200 predicted wells** per region  
- Applies **bootstrapping** to estimate average profit, confidence intervals, and loss probability  
- Recommends the region with the strongest stability and investment feasibility  

---

## Key Insights
- **Region 1** produced the lowest RMSE (0.89) and the most consistent predictions  
- **Regions 0 and 2** showed high variability (RMSE about 37.7 and 40.1)  
- Minimum reserve volume to avoid loss: **111.1 thousand barrels**  
- Bootstrapping confirmed Region 1 offers the most reliable profit and the least downside risk  

---

## Results
| Region   | RMSE  | Risk of Loss  | Profitability      |
|----------|-------|---------------|--------------------|
| Region 0 | 37.69 | Moderate      | Variable           |
| **Region 1** | **0.89** | **Low** | **Most Profitable** |
| Region 2 | 40.08 | Moderate-High | Unstable           |

---

## Tech Stack
**Python**, Pandas, NumPy, Scikit-learn, Matplotlib, Streamlit  

---

## Installation
```bash
# Clone repository
git clone https://github.com/rhicarmel/oil-well-prediction.git
cd oil-well-prediction

# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/OilWells.ipynb

# Or run the Streamlit app
streamlit run app/streamlit_app.py
```
---

## Future Improvements
- Add **ensemble models** such as Random Forest or XGBoost to improve prediction stability  
- Incorporate **geological coordinate visualizations** for spatial analysis of well clusters  
- Expand **profit simulations** to include operational, market, and variability-based cost scenarios  
- Build a **Streamlit dashboard** for interactive region comparison and forecast exploration  

---

## Author
**Rhiannon Fillingham**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/rhiannonfilli)
