import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# ================================
# 1. Télécharger les données
# ================================

ticker = "COV.PA"
benchmark = "^FCHI"  # CAC 40

start_date = "2015-01-01"
end_date = None

cov_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
bench_data = yf.download(benchmark, start=start_date, end=end_date, auto_adjust=True)

# On utilise la colonne Close et on force en Series
cov_prices = cov_data["Close"].dropna()
bench_prices = bench_data["Close"].dropna()

# Sécurité si jamais yfinance renvoie un DataFrame
if isinstance(cov_prices, pd.DataFrame):
    cov_prices = cov_prices.iloc[:, 0]
if isinstance(bench_prices, pd.DataFrame):
    bench_prices = bench_prices.iloc[:, 0]

# ================================
# 2. Alignement des séries & rendements
# ================================

# On met tout dans un DataFrame pour aligner sur les mêmes dates
df = pd.concat([cov_prices, bench_prices], axis=1, join="inner")
df.columns = ["cov", "bench"]

# Rendements journaliers
returns = df.pct_change().dropna()
cov_ret = returns["cov"]
bench_ret = returns["bench"]

# Si jamais il n'y a pas assez de données, on stoppe
if len(cov_ret) == 0 or len(bench_ret) == 0:
    raise ValueError("Pas de rendements après alignement. Vérifie les données téléchargées.")

# ================================
# 3. BETA
# ================================

# Beta = Cov(r_cov, r_bench) / Var(r_bench)
cov_arr = cov_ret.values
bench_arr = bench_ret.values

covariance = np.cov(cov_arr, bench_arr)[0, 1]
variance_bench = np.var(bench_arr)

beta = covariance / variance_bench

# ================================
# 4. Max Drawdown
# ================================

# On réutilise la série de prix alignée pour Covivio
cov_prices_aligned = df["cov"]

running_max = cov_prices_aligned.cummax()
drawdown = cov_prices_aligned / running_max - 1.0  # en -xx%

max_drawdown = drawdown.min()  # valeur la plus creuse (négative)

# ================================
# 5. Skewness & Kurtosis
# ================================

skewness = skew(cov_arr, bias=False)
kurt_val = kurtosis(cov_arr, fisher=False, bias=False)  # 3 = normale

# ================================
# 6. Affichage des résultats
# ================================

print("\n===== METRIQUES COVIVIO =====")
print(f"Beta vs CAC40       : {beta:.4f}")
print(f"Max Drawdown        : {max_drawdown*100:.2f}%")
print(f"Skewness            : {skewness:.4f}")
print(f"Kurtosis            : {kurt_val:.4f} (3 = distribution normale)")

# ================================
# 7. Graphique prix Covivio vs CAC40
# ================================

plt.figure(figsize=(12, 6))

# Normalise le CAC40 pour partir du même point que Covivio
bench_norm = df["bench"] / df["bench"].iloc[0] * df["cov"].iloc[0]

plt.plot(df.index, df["cov"], label="Covivio (COV.PA)")
plt.plot(df.index, bench_norm, label="CAC40 (normalisé)")

plt.title("Covivio vs CAC40 (prix normalisés)")
plt.xlabel("Date")
plt.ylabel("Prix")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ================================
# 8. Graphique drawdown Covivio
# ================================

plt.figure(figsize=(12, 5))
plt.plot(drawdown.index, drawdown.values, color="red")
plt.title("Drawdown de Covivio")
plt.xlabel("Date")
plt.ylabel("Drawdown (en proportion)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ================================
# 9. Graphique distribution des rendements (Skewness & Kurtosis)
# ================================

import scipy.stats as stats

plt.figure(figsize=(10,6))

# Histogramme
plt.hist(cov_ret, bins=50, density=True, alpha=0.6, color='skyblue', label="Rendements Covivio")

# Courbe de la loi normale théorique
mean = cov_ret.mean()
std = cov_ret.std()
x = np.linspace(mean - 4*std, mean + 4*std, 500)
normal_pdf = stats.norm.pdf(x, mean, std)

plt.plot(x, normal_pdf, 'r', linewidth=2, label="Loi normale (μ, σ)")

# Titre & légende
plt.title("Distribution des rendements Covivio\nSkewness & Kurtosis")
plt.xlabel("Rendements journaliers")
plt.ylabel("Densité")
plt.grid(True)

# Indicateurs affichés directement sur le graphique
plt.text(x.min(), normal_pdf.max()*0.9,
         f"Skewness : {skewness:.4f}\nKurtosis : {kurt_val:.4f}",
         fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

plt.legend()
plt.tight_layout()
plt.show()

