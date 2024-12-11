import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Leer el archivo CSV
data = pd.read_csv("sales_data.csv")

# Verificar datos iniciales
print("Primeras filas del dataset:")
print(data.head())
print("\nVerificación de valores nulos:")
print(data.isnull().sum())

# Separar características (X) y variable objetivo (y)
X = data.drop(columns=["Profit", "Product"])  # Eliminar 'Profit' y 'Product'
y = data["Profit"]

print("\nCaracterísticas (X):")
print(X.head())
print("\nVariable objetivo (y):")
print(y.head())

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X_scaled, y)

# Hacer predicciones
y_pred = model.predict(X_scaled)

# Evaluar el modelo
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\nEvaluación del modelo:")
print(f"Error Cuadrático Medio (MSE): {mse:.4f}")
print(f"Coeficiente de Determinación (R²): {r2:.4f}")

# Visualización: Comparación entre valores reales y predichos
plt.figure(figsize=(10, 6))
plt.plot(range(len(y)), y, label="Valores Reales", marker='o')
plt.plot(range(len(y_pred)), y_pred, label="Predicciones", marker='x')
plt.title("Comparación entre Valores Reales y Predicciones")
plt.xlabel("Índice de la Muestra")
plt.ylabel("Profit")
plt.legend()
plt.grid()
plt.show()

# Visualización: Residuales
residuals = y - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(range(len(residuals)), residuals, color='purple')
plt.axhline(0, color='red', linestyle='--')
plt.title("Distribución de los Residuales")
plt.xlabel("Índice de la Muestra")
plt.ylabel("Residual")
plt.grid()
plt.show()