# Relación carga/masa del electrón

Análisis de datos de la práctica de laboratorio e/m (tubo de haz filiforme + bobinas de Helmholtz).

## Ejecutar

```bash
pip install -r requirements.txt
python analisis.py
```

Los datos están en `datos.py`. Las figuras quedan en `figuras/` y las tablas/resultados en `resultados/`.

## Método

1. **Caracterización de la bobina**: regresión lineal `B = m I + b` con los datos de campo (ya corregidos por el campo ambiental `B_amb = (0,04 ± 0,01) mT`).
2. **Campo para cada punto**: `B = m I + b`, con incertidumbre propagada
   `σ_B² = I² σ_m² + σ_b² + 2 I cov(m,b) + m² σ_I²` y `σ_{B²} = 2 B σ_B`.
3. **Para cada radio**: regresión `V = s B² + c` (gráficas V vs B² y V vs I con la curva `V = s (m I + b)² + c`).
4. **Relación carga/masa**: `e/m = 2 s / r²`, con `σ_{e/m}² = (2 σ_s / r²)² + (4 s σ_r / r³)²`.

Las regresiones usan mínimos cuadrados ponderados con varianza efectiva
(`σ_ef² = σ_y² + (a σ_x)²`) para considerar las incertidumbres en ambos ejes.
Si `χ²_red > 1`, las incertidumbres de los parámetros se escalan por `√χ²_red`.

Incertidumbres: `V ± 1 V`, `I ± 0,01 A`, `B ± 0,01 mT`. La incertidumbre del radio (`SIGMA_R` en `datos.py`) está en 0; cámbiala si la conoces.

Los puntos excluidos del ajuste se configuran en `EXCLUIR` dentro de `analisis.py`.
