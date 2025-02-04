import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# Create directories for saving the graphs and period datasets
os.makedirs('graphs', exist_ok=True)
os.makedirs('dataset_periods', exist_ok=True)

# Load the dataset
output_file = 'dataset_weekly_covid_19_italy.csv'
data = pd.read_csv(output_file, parse_dates=['start_of_week', 'data_end_week'])

# Define start reference date
start_reference = pd.to_datetime('2021-09-23')  # Data di inizio corretta

# Calculate the week number
data['week_number'] = ((data['start_of_week'] - start_reference).dt.days // 7).astype(int)
min_week = data['week_number'].min()
data['week_number'] = data['week_number'] - min_week

# Chosen Periods
periods = [
    ('2021-09-23', '2022-01-28', 'Winter 2021'),
    ('2022-02-04', '2022-06-10', 'Intermediate 2022'),
    ('2022-06-17', '2022-08-19', 'Summer 2022'),
    ('2022-08-26', '2022-12-23', 'Winter 2022'),
    ('2022-12-30', '2023-06-30', 'Intermediate 2023'),
    ('2023-07-07', '2023-09-07', 'Summer 2023'),
    ('2023-09-14', '2023-12-28', 'Winter 2023'),
    ('2024-01-04', '2024-05-23', 'Intermediate 2024'),
    ('2024-05-30', '2024-09-19', 'Summer 2024')
]

# Function to filter the data based on the period
def filter_period(data, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    mask = (data['start_of_week'] >= start_date) & (data['start_of_week'] < end_date + pd.Timedelta(days=1))
    return data.loc[mask].copy()

# Function to apply linear regression
def apply_regression(X, Y):
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    return model

# Graphs colors
colors = ['#1f77b4', '#808080', '#d62728', '#1f78b4', '#8C9E8E', '#ff7f0e', '#08306b', '#8C8279', '#8c564b', '#9467bd']

# Create the overall graph without highlighted points but with background colors
plt.figure(figsize=(14, 10))

ax = plt.gca()  # Get the current axes

x_starts = []

# Plot the data for each period
for i, (start_date, end_date, period_name) in enumerate(periods):
    period_data = filter_period(data, start_date, end_date)

    if period_data.empty:
        continue

    period_data.to_csv(os.path.join('dataset_periods', f'dataset_{period_name}.csv'), index=False)

    X = period_data['week_number']
    Y = period_data['deceduti_diff']

    x_starts.append(X.min())
    plt.scatter(X, Y, color=colors[i], alpha=0.6, s=10, label=f'Data {period_name}')

x_starts.append(data['week_number'].max())

for i in range(len(x_starts) - 1):
    ax.axvspan(x_starts[i], x_starts[i + 1], color=colors[i], alpha=0.1)

plt.xlabel('Weeks Since First Measurement')
plt.ylabel('Weekly Deaths')
plt.title('Overall Graph')
plt.legend()
plt.grid(True)

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlim(0, data['week_number'].max())

# Margins for better visualization
plt.margins(x=0.05, y=0.1)

# Save the graph without regression lines
plt.savefig(os.path.join('graphs', 'overall_graph_without_regression.png'))
plt.show()

# Add regression lines to the overall graph
plt.figure(figsize=(14, 10))

ax = plt.gca()  # Get the current axes

for i, (start_date, end_date, period_name) in enumerate(periods):
    period_data = filter_period(data, start_date, end_date)

    if period_data.empty:
        continue

    X = period_data['week_number']
    Y = period_data['deceduti_diff']

    plt.scatter(X, Y, color=colors[i], alpha=0.6, s=10, label=f'Data {period_name}')

    # Apply regression and plot the line
    model = apply_regression(X, Y)
    X_vals = np.linspace(X.min(), X.max(), 100)
    Y_vals = model.predict(sm.add_constant(X_vals))
    plt.plot(X_vals, Y_vals, color=colors[i], linestyle='-', linewidth=2)

# Add background spans for periods
for i in range(len(x_starts) - 1):
    ax.axvspan(x_starts[i], x_starts[i + 1], color=colors[i], alpha=0.1)

plt.xlabel('Weeks Since First Measurement')
plt.ylabel('Weekly Deaths')
plt.title('Overall Graph with Regression Lines')
plt.legend()
plt.grid(True)

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlim(0, data['week_number'].max())

# Margins for better visualization
plt.margins(x=0.05, y=0.1)

# Save the graph with regression lines
plt.savefig(os.path.join('graphs', 'overall_graph_with_regression.png'))
plt.show()

# Create the regression results file
with open('regression_results.txt', 'w') as results_file:
    results_file.write('Period, Slope, Intercept, R^2, P-value, Mean, Std Dev, Variance\n')
    cumulative_week_start = 0
    # Loop through each period
    for i, (start_date, end_date, period_name) in enumerate(periods):
        period_data = filter_period(data, start_date, end_date)

        if period_data.empty:
            continue

        # Calculate the relative week number
        period_start = pd.to_datetime(start_date)
        period_data.loc[:, 'relative_week_number'] = ((period_data['start_of_week'] - period_start).dt.days // 7).astype(int)

        # Check if there are duplicate weeks and increment the week number if so
        seen_weeks = set()
        for idx, week in period_data['relative_week_number'].items():
            if week in seen_weeks:
                # Increment the week number if it has been seen before
                period_data.at[idx, 'relative_week_number'] += 1
            seen_weeks.add(period_data.at[idx, 'relative_week_number'])

        # Update the minimum week number
        min_week = period_data['relative_week_number'].min()
        period_data.loc[:, 'relative_week_number'] = period_data['relative_week_number'] - min_week

        # Apply linear regression
        X_relative = period_data['relative_week_number']  # Use relative number for the model
        X_cumulative = X_relative + cumulative_week_start  # Use cumulative number for plotting
        Y = period_data['deceduti_diff']
        model = apply_regression(X_relative, Y)

        X_vals_week_number = np.arange(X_relative.min(), X_relative.max() + 1)
        X_vals = pd.DataFrame({'const': 1, 'week_number': X_vals_week_number})
        Y_vals = model.predict(X_vals)

        # Plot the data and regression line
        plt.figure(figsize=(12, 8))
        plt.scatter(X_cumulative, Y, color=colors[i], alpha=0.6, label=f'Data {period_name}', s=10)
        plt.plot(X_vals_week_number + cumulative_week_start, Y_vals, label=f'Regression {period_name}', color=colors[i], linestyle='-', linewidth=2)

        plt.xlabel('Cumulative Week Number')
        plt.ylabel('Weekly Deaths')
        plt.title(f'Linear Regression for the Period {period_name} ({start_date} - {end_date})')
        plt.legend()
        plt.grid(True)

        # Forcing the x-axis to be integer values
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        # X axis limits
        plt.xlim(cumulative_week_start, cumulative_week_start + X_relative.max())  # Limiti aggiornati per l'asse X

        # Margins for better visualization
        plt.margins(x=0.05, y=0.1)

        # Save the graph
        plt.savefig(os.path.join('graphs', f'graph_{period_name}_{start_date}_to_{end_date}.png'))
        plt.close()

        # Print the results
        slope = model.params.iloc[1]  # Slope
        intercept = model.params.iloc[0]  # Intercept
        r_squared = model.rsquared  # R^2
        p_value = model.pvalues.iloc[1]  # P-value

        results_file.write(f"{period_name}, {slope:.4f}, {intercept:.4f}, {r_squared:.4f}, {p_value:.4f}, {Y.mean():.4f}, {Y.std():.4f}, {Y.var():.4f}\n")

        print(f"\n--- Results for {period_name} ({start_date} - {end_date}) ---")
        print(f"Slope: {slope:.4f}")
        print(f"Intercept: {intercept:.4f}")
        print(f"R^2: {r_squared:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Mean Deaths: {Y.mean():.4f}")
        print(f"Standard Deviation of Deaths: {Y.std():.4f}")
        print(f"Variance of Deaths: {Y.var():.4f}")
        print("------------------------------------------------------\n")

        # Update the cumulative week start
        cumulative_week_start += X_relative.max() + 1

 # Read file and calculate the mean of R^2 for each season
results_file = 'regression_results.txt'
results_df = pd.read_csv(results_file, delimiter=',')

# Remove spaces
results_df.columns = results_df.columns.str.strip()

# Extract the season from the period name and calculate the mean of R^2 for each season
results_df['Season'] = results_df['Period'].str.split(' ').str[0]  # Estrai la prima parola (stagione)
seasonal_r2_mean = results_df.groupby('Season')['R^2'].mean().reset_index()

# Save the results to a CSV file
seasonal_r2_mean.to_csv('seasonal_r2_means.csv', index=False)

# Print the results
print("\n--- R^2 mean for every season---")
print(seasonal_r2_mean)