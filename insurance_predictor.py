import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import sys,re

def convert_age_to_months(age_str):
    age_str = str(age_str)  # Ensure the input is treated as a string
    years = 0
    months = 0
    if 'years' not in age_str:
        years = 0
        months = int(age_str.split()[0])
    else:
        years = int(age_str.split()[0])
        months = int(age_str.split()[3])
    return years * 12 + months

def encode_ages(df):
    # Encode ages 23 to 78 as classes 0 to 55
    df['age_class'] = df['age_of_policyholder'] - 23
    return df


def process_data_fortrain(df):
    # Create a copy of the DataFrame to avoid modifying the original data
    df_copy = df.copy()

    # Binary feature conversion
    binary_features = [col for col in df_copy.columns if df_copy[col].dropna().isin(['Yes', 'No']).all()]
    for col in binary_features:
        df_copy[col] = df_copy[col].map({'Yes': 1, 'No': 0})

    # List of numeric features for normalization
    numeric_features = ['policy_tenure', 'population_density', 'airbags', 'displacement', 'cylinder', 'turning_radius', 'length', 'width', 'height', 'gross_weight']

    # Categorical feature encoding
    categorical_features = ['area_cluster', 'segment', 'model', 'fuel_type', 'engine_type', 'rear_brakes_type', 'transmission_type', 'steering_type']
    for col in categorical_features:
        df_copy[col], _ = pd.factorize(df_copy[col])

    # Extract numeric values from 'max_torque' and 'max_power'
    df_copy['max_torque'] = df_copy['max_torque'].astype(str).str.extract(r'(\d+)').astype(float)
    df_copy['max_power'] = df_copy['max_power'].astype(str).str.extract(r'(\d+\.\d+|\d+)').astype(float)

    # Convert 'age_of_car' to total months using a previously defined function convert_age_to_months
    df_copy['age_of_car'] = df_copy['age_of_car'].apply(convert_age_to_months)

    # Normalize numeric features
    df_copy[numeric_features] = (df_copy[numeric_features] - df_copy[numeric_features].mean()) / df_copy[numeric_features].std()

    return df_copy


def process_data_fortest(df):
    # Create a copy of the DataFrame to avoid modifying the original data
    df_copy = df.copy()

    # Binary feature conversion
    binary_features = [col for col in df_copy.columns if df_copy[col].dropna().isin(['Yes', 'No']).all()]
    for col in binary_features:
        df_copy[col] = df_copy[col].map({'Yes': 1, 'No': 0})

    # List of numeric features for normalization
    numeric_features = ['policy_tenure', 'population_density', 'airbags', 'displacement', 'cylinder', 'turning_radius', 'length', 'width', 'height', 'gross_weight']

    # Categorical feature encoding
    categorical_features = ['area_cluster', 'segment', 'model', 'fuel_type', 'engine_type', 'rear_brakes_type', 'transmission_type', 'steering_type']
    for col in categorical_features:
        df_copy[col], _ = pd.factorize(df_copy[col])

    # Extract numeric values from 'max_torque' and 'max_power'
    df_copy['max_torque'] = df_copy['max_torque'].astype(str).str.extract(r'(\d+)').astype(float)
    df_copy['max_power'] = df_copy['max_power'].astype(str).str.extract(r'(\d+\.\d+|\d+)').astype(float)

    # Convert 'age_of_car' to total months using a previously defined function convert_age_to_months
    df_copy['age_of_car'] = df_copy['age_of_car'].apply(convert_age_to_months)

    # Normalize numeric features
    df_copy[numeric_features] = (df_copy[numeric_features] - df_copy[numeric_features].mean()) / df_copy[numeric_features].std()

    return df_copy


def train_and_evaluate(train_df, test_df):
    tmp_train_df = process_data_fortrain(train_df)
    tmp_test_df = process_data_fortrain(test_df)
    tmp_train_df = encode_ages(tmp_train_df)
    tmp_test_df = encode_ages(tmp_test_df)
    X = tmp_train_df.drop(['age_of_policyholder', 'is_claim', 'policy_id'], axis=1)
    X_test = tmp_test_df.drop(['age_of_policyholder', 'is_claim', 'policy_id'], axis=1, errors='ignore')
    # X = process_data_fortrain(X)
    # X_test = process_data_fortest(X_test)
    features = ['policy_tenure', 'age_of_car', 'area_cluster', 'population_density']

    # print(X)
    y_age = tmp_train_df['age_class']
    y_test = tmp_test_df['age_class']
    y_claim = tmp_train_df['is_claim']
    y_claim_test = tmp_test_df['is_claim']

    class_weights = {0: 1, 1: 10}

    

    models = {
        # 'Ridge': Ridge(alpha = 0.2),
        # 'Lasso': Lasso(),
        # 'ElasticNet': ElasticNet(),
        # 'DecisionTree': DecisionTreeRegressor(max_depth = 10,min_samples_split=50, min_samples_leaf=10),
        # 'RandomForest': RandomForestRegressor(n_estimators = 70, max_depth = 10,min_samples_split=50, min_samples_leaf=10),
        # 'mlp': MLPRegressor(hidden_layer_sizes=(400,100), max_iter=100, activation='relu', verbose=True)
        'RandomForest': RandomForestClassifier(),
        # 'GradientBoosting': GradientBoostingRegressor(n_estimators = 60, max_depth = 10, learning_rate = 0.08, min_samples_split=55, min_samples_leaf=10)
   #n_estimators = 75~120, max_depth = 8-15, learning_rate = 0.07 - 0.15, min_samples_split=~55理解55正解, min_samples_leaf=10~20偏不重要
    }

    # param_grid = {
    # 'n_estimators': [40, 50, 70, 80, 100, 200, 300],
    # 'learning_rate': [0.001, 0.05, 0.01, 0.1, 0.2],
    # 'max_depth': [3, 4, 5, 6],
    # 'min_samples_split': [2, 4],
    # 'min_samples_leaf': [1, 2],
    # 'subsample': [0.8, 0.9, 1.0]
    # }
    
    # pca = PCA(n_components=40)
    # X = pca.fit_transform(X)
    # X_test = pca.transform(X_test)
    # print(X.shape)
    
  
    results = {}
    for name, model in models.items():
        # grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
        # grid_search.fit(X, y_age)
        model.fit(X,y_age)
        age_pred_train = model.predict(X)
        age_pred_test = model.predict(X_test)


        # Convert predictions back to ages
        age_pred_train = age_pred_train+23
        age_pred_test = age_pred_test+23
        age_pred_test = np.round(age_pred_test).astype(int)
        print(age_pred_train.max())
        print(age_pred_train.min())
        # print(test_df['age_of_policyholder'])
        # print(age_pred_test)
        print(age_pred_test.max())
        print(age_pred_test.min())
        mse_train = mean_squared_error(train_df['age_of_policyholder'], age_pred_train)
        mse_test = mean_squared_error(test_df['age_of_policyholder'], age_pred_test)
        results[name] = {'Train MSE': mse_train, 'Test MSE': mse_test}
        print(name)
        print(f"Train MSE: {mse_train}")
        print(f"Test MSE: {mse_test}")

    forest = RandomForestRegressor()
    forest.fit(X, y_age)
    importances = forest.feature_importances_
    # print(importances)
    feature_importances = pd.DataFrame({
        'feature': X.columns,   # Assuming X is a DataFrame
        'importance': importances
    })



    # Sort the DataFrame by importance
    feature_importances_sorted = feature_importances.sort_values(by='importance', ascending=False)
    top_features = feature_importances_sorted.head(10)['feature']
    X = X[top_features]
    X_test = X_test[top_features]
    
    rf_claim = RandomForestClassifier(class_weight=class_weights,max_depth = 7,random_state = 1)
    rf_claim.fit(X, y_claim)
    claim_pred_train =  rf_claim.predict(X)
    claim_pred_test =  rf_claim.predict(X_test) 
    f1_train = f1_score(y_claim, claim_pred_train, average='macro')
    f1_test = f1_score(y_claim_test, claim_pred_test, average='macro')
    print(f"Train Macro F1-Score: {f1_train}")
    print(f"Test Macro F1-Score: {f1_test}")

    return age_pred_test, claim_pred_test, test_df['policy_id'], mse_train, f1_train, mse_test, f1_test

def write_output(age_predictions, claim_predictions, policy_ids, id):
    part1_output = pd.DataFrame({
        'policy_id': policy_ids,
        'age': age_predictions
    })
    part1_output.to_csv(f'z{id}.PART1.output.csv', index=False)

    part2_output = pd.DataFrame({
        'policy_id': policy_ids,
        'is_claim': claim_predictions
    })
    part2_output.to_csv(f'z{id}.PART2.output.csv', index=False)


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 z{id}.py train.csv test.csv")
        sys.exit(1)

    script_name = sys.argv[0]
    train_file, test_file = sys.argv[1], sys.argv[2]

    # Extract the id from the script filename using regular expression
    match = re.search(r'z(\d+)\.py', script_name)
    if not match:
        print("Error: Script name must be in the format z{id}.py where {id} is a number.")
        sys.exit(1)

    id = match.group(1)  # Extract the id part

    # Load data
    train_df = pd.read_csv('train.csv', usecols=list(range(1, 45)))
    test_df = pd.read_csv('test.csv', usecols=list(range(1, 45)))


    # Process data and evaluate model
    age_predictions, claim_predictions, policy_ids, mse_train, f1_train, mse_test, f1_test= train_and_evaluate(train_df, test_df)

    # Write output files
    write_output(age_predictions, claim_predictions, policy_ids, id)

if __name__ == "__main__":
    main()
