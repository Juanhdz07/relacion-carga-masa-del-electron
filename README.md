# Relación carga/masa del electrón

Gráficas de la práctica de laboratorio e/m (tubo de haz filiforme + bobinas de Helmholtz).

## Ejecutar

```bash
pip install -r requirements.txt
python graficas.py
```

Los datos están en `datos.py` y las figuras quedan en `figuras/`:

- `B_vs_I.png`: caracterización de la bobina, regresión `B = m I + b`.
- `V_vs_B2_r{2,4,6,8}cm.png`: regresión por el origen `V = m' B²` para cada radio.

## Método

1. **Caracterización de la bobina**: regresión lineal `B = m I + b` con los datos de campo (ya corregidos por el campo ambiental `B_amb = (0,04 ± 0,01) mT`).
2. **Campo para cada punto**: `B = m I + b`, con incertidumbre propagada
   `σ_B² = I² σ_m² + σ_b² + 2 I cov(m,b) + m² σ_I²` y `σ_{B²} = 2 B σ_B`.
3. **Para cada radio**: regresión por el origen `V = m' B²`, de donde `e/m = 2 m' / r²`.

Las regresiones usan mínimos cuadrados ponderados con varianza efectiva
(`σ_ef² = σ_y² + (a σ_x)²`) para considerar las incertidumbres en ambos ejes.
Si `χ²_red > 1`, las incertidumbres de los parámetros se escalan por `√χ²_red`.

Incertidumbres: `V ± 1 V`, `I ± 0,01 A`, `B ± 0,01 mT`.

Los puntos excluidos del ajuste se configuran en `EXCLUIR` dentro de `graficas.py`.
