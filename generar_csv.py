import pandas as pd

# Crear un DataFrame de ejemplo
data = {
    "Product": ["P1", "P2", "P3", "P4", "P5"],
    "Sales_Q1": [11, 7, 7, 12, 8],
    "Sales_Q2": [12, 6, 11, 8, 5],
    "Sales_Q3": [10, 3, 8, 13, 13],
    "Sales_Q4": [8, 2, 9, 5, 11],
    "Profit": [0.28, 0.50, 0.18, 0.06, 0.27]
}

# Convertir a DataFrame
df = pd.DataFrame(data)

# Guardar como archivo CSV
df.to_csv("sales_data.csv", index=False)

print("Archivo 'sales_data.csv' generado con éxito.")
