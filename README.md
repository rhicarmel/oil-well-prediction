# 🛢️ OIL WELL PROFITABILITY PREDICTION (REGIONAL ANALYSIS)

## Overview
A machine learning project designed to identify the **most profitable and least risky region** for new oil well development.  
Using geological data from three potential regions, the model predicts oil reserves and evaluates profit, risk, and investment feasibility.

**Goal:** Select the optimal region for drilling based on expected profit and risk analysis.  
**Best Region:** 🏆 Region 1 — highest average profit, lowest RMSE, and minimal loss risk.

🔗 [View the full notebook here](./OilWells.ipynb)

[▶️ Launch Interactive App](https://deepnote.com/app/projects-8f6f/Optimal-Oil-Well-Development-Region-Selection-Tool-c1e8aaa2-86aa-404a-b687-fae09fa91ce9?utm_content=c1e8aaa2-86aa-404a-b687-fae09fa91ce9&__run=true)

---

## Functionality
- Loads and analyzes geological data from three oil regions.  
- Trains a **Linear Regression model** to predict oil reserves.  
- Calculates **RMSE** for model accuracy per region.  
- Simulates profit from the **top 200 predicted wells** per region.  
- Uses **bootstrapping** to assess average profit, confidence intervals, and loss probability.  
- Recommends the best region for drilling based on profitability and stability.

---

## Key Insights
- **Region 1** demonstrated the lowest RMSE (**0.89**) and most consistent predictions.  
- **Regions 0 and 2** had higher RMSEs (≈ 37.7 and 40.1), indicating greater variability.  
- The **minimum viable reserve** to avoid loss was **111.1 thousand barrels**.  
- Bootstrapping analysis confirmed **Region 1** offers the **highest average profit** with the **lowest financial risk**.  

---

## Results
| Region | RMSE | Risk of Loss | Profitability |
|---------|------|--------------|----------------|
| Region 0 | 37.69 | Moderate | Variable |
| **Region 1** | **0.89** | **Low** | **Most Profitable** |
| Region 2 | 40.08 | Moderate–High | Unstable |

---

## Tech Stack
**Python**, Pandas, NumPy, Scikit-learn, Matplotlib  
*Developed in Jupyter Notebook*

---

## Installing
```bash
# Clone repo
git clone https://github.com/rhicarmel/oil-well-prediction.git
cd oil-well-prediction

# Install dependencies
pip install -r requirements.txt

# Open notebook
jupyter notebook OilWells.ipynb
```
---

## Future Improvements
- Implement **ensemble learning methods** (Random Forest, XGBoost) to improve prediction stability.  
- Include **geological coordinate mapping** for visual exploration of oil well clusters.  
- Enhance **profit simulation** by incorporating operational and market costs.  
- Develop a **dashboard or API** for interactive region comparison and forecast visualization.  

---

## Author
**Rhiannon Fillingham**  
📎 [LinkedIn](https://www.linkedin.com/in/rhiannonfilli)
