import warnings
import numpy as np
import pandas as pd
import random
from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

warnings.filterwarnings("ignore", category=RuntimeWarning, module="deap.creator")


DATA_URL = "https://raw.githubusercontent.com/Athulyachandran/House-Rent-Dataset-Analysis/main/House_Rent_Dataset.csv"

df = pd.read_csv(DATA_URL)

df.columns = df.columns.str.strip()

q_high = df['Rent'].quantile(0.99)
df = df[df['Rent'] <= q_high]

def parse_floor(floor_str):
    try:
        parts = str(floor_str).split(' out of ')
        level_str = parts[0].strip().lower()
        
        if 'ground' in level_str:
            level = 0
        elif 'lower basement' in level_str:
            level = -2
        elif 'upper basement' in level_str:
            level = -1
        else:
            level = int(level_str)
            
        total = int(parts[1]) if len(parts) > 1 else level
        return pd.Series([level, total])
    except:
        return pd.Series([1, 1])

df[['Floor_Level', 'Total_Floors']] = df['Floor'].apply(parse_floor)

categorical_cols = ['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

drop_cols = ['Posted On', 'Floor', 'Area Locality', 'Rent']
X_df = df_encoded.drop(columns=drop_cols)
y = df_encoded['Rent'].values

feature_names = X_df.columns.tolist()
X = X_df.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
num_features = X_train.shape[1]

print(f"Total encoded features available for GA: {num_features}")

if hasattr(creator, "FitnessMax"):
    del creator.FitnessMax
if hasattr(creator, "Individual"):
    del creator.Individual

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
def evaluate_feature_subset(individual):
    selected_indices = [i for i, bit in enumerate(individual) if bit == 1]
    
    if len(selected_indices) == 0:
        return (-1.0,)
    
    X_tr_sub = X_train[:, selected_indices]
    X_te_sub = X_test[:, selected_indices]
    
    fast_model = Ridge(alpha=100.0)
    fast_model.fit(X_tr_sub, y_train)
    
    preds = fast_model.predict(X_te_sub)
    r2 = r2_score(y_test, preds)
    return (r2,)

toolbox.register("evaluate", evaluate_feature_subset)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def run_fast_ga():
    random.seed(42)
    pop = toolbox.population(n=50)
    
    pop, logbook = algorithms.eaSimple(
        pop, toolbox, 
        cxpb=0.6, 
        mutpb=0.2, 
        ngen=20, 
        verbose=False
    )
    
    best_ind = tools.selBest(pop, 1)[0]
    selected_indices = [i for i, bit in enumerate(best_ind) if bit == 1]
    selected_features = [feature_names[i] for i in selected_indices]
    
    print(f"\n[GA Complete] Selected {len(selected_indices)} / {num_features} Features:")
    for feat in selected_features:
        print(f"  [✓] {feat}")

    final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    final_model.fit(X_train[:, selected_indices], y_train)
    
    y_pred = final_model.predict(X_test[:, selected_indices])
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2 = r2_score(y_test, y_pred)
    
    print("\n================ FINAL MODEL RESULTS ================")
    print(f"  - R² Score:                         {r2:.4f}")
    print(f"  - RMSE (Currency):                 ₹{rmse:,.2f}")
    print(f"  - MAE (Mean Absolute Error):        ₹{mae:,.2f}")
    print(f"  - MAPE (Mean Absolute % Error):    {mape:.2f}%")
    
    print("\nSample Rent Predictions vs. Actual Rent:")
    for i in range(5):
        print(f"  Property {i+1}: Actual = ₹{y_test[i]:,.0f} | GA Predicted = ₹{y_pred[i]:,.2f}")

if __name__ == "__main__":
    run_fast_ga()
