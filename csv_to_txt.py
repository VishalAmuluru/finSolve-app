import pandas as pd

# Load the CSV
df = pd.read_csv("data/hyderabad_loan_offers_10k.csv")

# Create data.txt with formatted lines
with open("data.txt", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        line = (
            f"Bank: {row['bank_name']}, "
            f"Loan Type: {row['loan_type']}, "
            f"Interest Rate: {row['interest_rate']}%, "
            f"Tenure: {row['tenure_years']} years, "
            f"Min Amount: ₹{row['min_amount']}, "
            f"Max Amount: ₹{row['max_amount']}, "
            f"Processing Fee: {row['processing_fee']}, "
            f"Employment: {row['employment_type']}, "
            f"Location: {row['location']}, "
            f"Note: {row['description']}\n"
        )
        f.write(line)
