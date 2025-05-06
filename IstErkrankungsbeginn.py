import pandas as pd

# Load the CSV file
df = pd.read_csv("CovidIstErkrankungsbeginn.csv")

print("Unique values before filtering:", df["IstErkrankungsbeginn"].unique())

# Convert column to integer (if stored as string/object)
df["IstErkrankungsbeginn"] = df["IstErkrankungsbeginn"].astype(int)

# Drop rows with missing values (optional, if needed)
df = df.dropna(subset=["IstErkrankungsbeginn"])

# Filter rows where IstErkrankungsbeginn is EXACTLY 1
filtered_df = df[df["IstErkrankungsbeginn"] == 1]

# Save the filtered data
filtered_df.to_csv("filtered_data_1_only.csv", index=False)

# Verify results
print("\nFiltered Data Summary:")
print(f"Total rows retained: {len(filtered_df)}")


# Filter rows where IstErkrankungsbeginn = 1
#filtered_data = df[df["IstErkrankungsbeginn"] == 1]

# Save the result (optional)
#filtered_data.to_csv("filtered_results.csv", index=False)

# Display the first 5 rows of the filtered data
#print(filtered_data.head())