# InternPe-Car-Price-Prediction-using-Machine-learning
A Machine learning project that predicts used car prices based on various features like company, model, year, kilometers driven, and fuel type.
## 📋 Project Overview

This project involves building a linear regression model to predict used car prices using data from Quikr. The model achieves **92% accuracy** (R2 score) after comprehensive data cleaning and preprocessing.

## ✨ Features

- **Data Cleaning & Preprocessing**: Handled inconsistent data, missing values, and data type conversions
- **Exploratory Data Analysis**: Visualized relationships between features and target variable
- **Machine Learning Model**: Linear Regression with One-Hot Encoding for categorical variables
- **Model Pipeline**: Integrated preprocessing and modeling in a single pipeline
- **Model Persistence**: Saved trained model using pickle for future predictions

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib & Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development

## 📊 Dataset Features

- **name**: Car model name
- **company**: Manufacturer company
- **year**: Manufacturing year
- **Price**: Target variable (price in INR)
- **kms_driven**: Kilometers driven by the car
- **fuel_type**: Type of fuel (Petrol/Diesel)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Car-Price-Prediction-ML.git
   cd Car-Price-Prediction-ML
```

1. Install required dependencies
   ```bash
   pip install -r requirements.txt

## 💻 Usage

Running the Jupyter Notebook

```bash
jupyter notebook
```

Open Project2.ipynb and run all cells to see the complete workflow.

Making Predictions

```python
import pickle
import pandas as pd
import numpy as np

# Load the trained model
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Make prediction
sample_data = pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],
                          data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5))

predicted_price = model.predict(sample_data)
print(f"Predicted Price: ₹{predicted_price[0]:,.2f}")
```

🔧 Project Workflow

1. Data Loading & Exploration
   · Loaded dataset from CSV file
   · Explored data structure and identified issues
2. Data Cleaning
   · Handled non-numeric values in 'year' column
   · Removed 'Ask For Price' entries
   · Cleaned 'kms_driven' by removing units and converting to numeric
   · Standardized car names by keeping first three words
   · Handled missing values in 'fuel_type'
3. Exploratory Data Analysis
   · Analyzed relationship between company and price
   · Examined impact of kilometers driven on price
   · Studied fuel type effect on pricing
   · Multivariate analysis combining company, fuel type, and year
4. Model Building
   · Feature selection and target variable separation
   · Train-test split with optimal random state
   · One-Hot Encoding for categorical variables
   · Linear Regression model implementation
   · Pipeline creation for streamlined processing
5. Model Evaluation & Deployment
   · Achieved 92% R2 score
   · Model serialization using pickle
   · Prediction functionality for new data

📈 Results

· Final Model Accuracy: 92% (R2 Score)
· Best Random State: 655
· Key Insights:
  · Newer cars with lower mileage command higher prices
  · Luxury brands (Audi, BMW, Mercedes) have higher price ranges
  · Diesel cars generally have different pricing patterns than petrol cars

👨‍💻 Author

Madiha Manzoor


📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check issues page.

🙏 Acknowledgments

· InternPe for the internship opportunity
· Quikr for the dataset
· Scikit-learn and Python communities for excellent documentation

