# Seasonal Agriculture Performance & Yield Optimization Analysis

An end-to-end data analytics, statistical testing, and predictive modeling pipeline designed to evaluate agricultural performance across cropping seasons (**Kharif**, **Rabi**, and **Zaid**). This project benchmarks yield productivity, resource utilization efficiency (water, soil, fertilizers), biological vulnerability (disease and pest pressures), and farm-level economic viability across Indian agricultural regions.

---

## 📌 Project Overview

Agriculture in seasonal climates faces severe variations in rainfall, ambient temperature, pest infestations, and input costs. This project conducts an empirical evaluation using a multi-regional agricultural dataset (4,000 farm observations across 28 parameters) to identify seasonal bottlenecks and optimize farm profitability.

### Key Objectives
* **Seasonal Benchmarking**: Quantify productivity (tonnes/ha), operational expenditures, and revenue margins across Kharif, Rabi, and Zaid seasons.
* **Resource Productivity**: Assess water-use efficiency (t / 1,000 m³) across traditional flood versus micro-irrigation systems (Drip and Sprinkler).
* **Statistical Significance**: Execute ANOVA and non-parametric Kruskal-Wallis tests to validate whether seasonal divergences in yield, profit, and pest risks are statistically significant.
* **Supervised Machine Learning**: Train an ensemble regression model (Random Forest Regressor) to predict crop yield and pinpoint primary environmental and input drivers.
* **Decision Support System**: Provide an interactive simulation tool for predictive scenario modeling and decision-making on farm scales.

---

## 🛠️ Tech Stack & Dependencies

* **Language**: Python 3.10+
* **Data Processing & Manipulation**: `pandas`, `numpy`
* **Statistical Analysis & Hypothesis Testing**: `scipy.stats` (One-Way ANOVA, Kruskal-Wallis H-test)
* **Machine Learning & Feature Evaluation**: `scikit-learn` (`RandomForestRegressor`, `train_test_split`, `r2_score`, `mean_squared_error`)
* **Data Visualization**: `matplotlib`, `seaborn`
* **Execution Environment**: Google Colaboratory / Jupyter Notebook

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

---

## 📊 Dataset Architecture

* **Observations**: 4,000 farm profiles
* **Features**: 28 columns spanning geographic, soil, climatic, operational, and financial dimensions

| Category | Primary Variables |
| :--- | :--- |
| **Identifiers & Geography** | `Farm_ID`, `State`, `District` |
| **Crop & Operational Metadata** | `Crop`, `Season` (Kharif, Rabi, Zaid), `Farm_Area_Hectares`, `Irrigation_Method` |
| **Environmental & Weather** | `Rainfall_mm`, `Avg_Temperature_C`, `Humidity_pct`, `Sunlight_Hours_Day` |
| **Soil & Biological Diagnostics** | `Soil_pH`, `Soil_Moisture_pct`, `Nitrogen_kg_ha`, `Phosphorus_kg_ha`, `Potassium_kg_ha`, `Disease_Pest_Risk_pct` |
| **Inputs & Agrochemicals** | `Fertilizer_kg_ha`, `Pesticide_Litre_ha`, `Seed_Quality_Score`, `Water_Used_m3` |
| **Production & Economics** | `Yield_Tonnes_Ha`, `Production_Tonnes`, `Market_Price_INR_Tonne`, `Total_Cost_INR`, `Revenue_INR`, `Profit_INR`, `Water_Efficiency_t_per_1000m3` |

---

## 🔬 Methodology & Pipeline

```
Raw CSV Dataset 
   │
   ├──> 1. Data Cleaning & Domain-Aware Imputation
   │       ├── Rainfall imputed by Season median
   │       ├── Soil Moisture imputed by [Season × Crop] median
   │       └── Yield calibrated: (Production / Farm Area) & group medians
   │
   ├──> 2. Feature Engineering
   │       ├── Profit Margin (%) = (Profit / Revenue) * 100
   │       ├── Input Normalization: Cost/Ha, Water Used (m³/Ha)
   │       └── Farm Scale Discretization: Small (<2 Ha), Medium (2–5 Ha), Large (>5 Ha)
   │
   ├──> 3. Exploratory Data Analysis & Statistical Testing
   │       ├── Parametric One-Way ANOVA
   │       ├── Non-Parametric Kruskal-Wallis tests
   │       └── Multi-seasonal Pearson correlation matrices
   │
   ├──> 4. Predictive Machine Learning Pipeline
   │       ├── One-Hot Encoding (Categorical variables)
   │       ├── 80/20 Train-Test split
   │       └── Random Forest Regressor (n_estimators=150, max_depth=12)
   │
   └──> 5. Scenario Simulation & Policy Formulation
           └── What-if engine for season, crop, irrigation, and area configuration
```

---

## 📈 Empirical Findings & Statistical Insights

### 1. Seasonal Benchmark Summary

| Metric | Kharif | Rabi | Zaid |
| :--- | :---: | :---: | :---: |
| **Sample Size (Farms)** | 1,779 | 1,627 | 594 |
| **Avg Rainfall (mm)** | 852.11 | 435.94 | 299.12 |
| **Avg Temperature (°C)** | 28.45 | 23.49 | 31.04 |
| **Avg Soil Moisture (%)** | 31.20 | 24.05 | 19.18 |
| **Avg Yield (Tonnes/Ha)** | **5.63** | 5.09 | 4.63 |
| **Water Efficiency (t/1,000 m³)** | **5.89** | 5.19 | 4.41 |
| **Pest & Disease Risk (%)** | **54.47%** | 40.48% | 38.22% |
| **Avg Net Profit (INR)** | **₹1,78,914.65** | ₹87,689.47 | **-₹24,804.82** |

### 2. Hypothesis Testing Results
* **Yield (t/ha)**: Statistically significant across seasons (Kruskal-Wallis $H = 70.6714$, $p = 4.51 \times 10^{-16}$).
* **Net Farm Profit (INR)**: Highly significant seasonal divergence (ANOVA $F = 34.29$, $p = 1.71 \times 10^{-15}$; Kruskal-Wallis $H = 101.93$, $p = 7.36 \times 10^{-23}$).
* **Water Used per Hectare (m³/ha)**: Not statistically significant ($p = 0.196$). Farms pump comparable volumes across seasons, but output varies drastically.
* **Disease & Pest Risk (%)**: Extreme seasonal divergence (ANOVA $F = 1049.47$, $p < 1.0 \times 10^{-300}$), peaking during humid Kharif monsoon months.

### 3. Irrigation Efficiency Comparison (t / 1,000 m³)

| Irrigation Method | Kharif | Rabi | Zaid |
| :--- | :---: | :---: | :---: |
| **Drip Irrigation** | 6.804 | 6.047 | 5.296 |
| **Sprinkler** | 4.622 | 4.644 | 4.860 |
| **Flood Irrigation** | 3.637 | 3.478 | 2.730 |
| **Rainfed** | 8.848 | 6.873 | 5.479 |

---

## 🤖 Machine Learning Model Performance

A **Random Forest Regressor** was trained to predict continuous crop yield (t/ha) using environmental attributes, soil nutrient densities (N, P, K), agrochemical usage, irrigation method, crop type, and seasonal indicators.

* **Coefficient of Determination ($R^2$)**: `0.9632`
* **Root Mean Squared Error (RMSE)**: `2.6624 Tonnes/Ha`

The feature importance rankings identify **crop species selection**, **seed quality score**, **water-use efficiency**, and **soil moisture availability** as the dominant determinants of yield output.

---

## 💡 Strategic Recommendations

1. **Mitigate Zaid Economic Losses**: Transition from water-intensive staples to drought-resilient short-duration pulses or oilseeds during Zaid. Avoid open flood irrigation during hot, dry spells.
2. **Preemptive Pest Management in Kharif**: Deploy Integrated Pest Management (IPM) protocols prior to monsoon onset to counter biological risks that average above 54%.
3. **Incentivize Micro-Irrigation Adoption**: Phase out flood irrigation in favor of subsidized drip systems, which boost water conversion efficiency from $2.73$ to $>5.29\text{ t}/1,000\text{ m}^3$ even in dry conditions.
4. **Smallholder Support Programs**: Smallholders (<2 Ha) experience the steepest margin compressions during off-peak dry cycles; targeted crop insurance and working capital credit are critical during Zaid.

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone [https://github.com/Ishwar2736/agriculture-vois-major-project/edit/main/README.md.git](
   
   ```
2. Place `seasonal_agriculture_performance_dataset.csv` in the project root directory.
3. Open and run the notebook:
   ```bash
   jupyter notebook Major_Project_VOIS.ipynb
   ```
4. Output artifacts (`seasonal_analysis_summary.csv` and `project_summary.json`) will be generated automatically in the working directory.
