# Demostraciones de Machine Learning

**Yuryen Rollo** · Machine Learning (202630_5646) · Universidad del Norte · Prof. Lihki Rubio Ortega

Este libro resuelve las demostraciones y verificaciones que quedaron como ejercicio en las
[notas del curso](https://lihkir.github.io/MachineLearning/intro.html). Cada capítulo enuncia el resultado, lo demuestra
paso a paso y, cuando es posible, lo comprueba numéricamente con Python.

| Capítulo | Tema | Qué se demuestra o verifica |
|---|---|---|
| 1 | K-NN | Las tres formas equivalentes del riesgo verdadero $L_{\mathcal D}(h)$ (ecuación 2.2), su relación con el riesgo empírico y la desigualdad $2\eta(1-\eta)\le 2\min\{\eta,1-\eta\}$ en el caso $\min=1-\eta$ |
| 2 | Regresión Ridge | $Z^T\mathbf 1=\mathbf 1^TZ=\mathbf 0$, las igualdades (3.11) y las cuatro propiedades (3.12) del estimador ridge |
| 3 | Regresión Lasso | Estimadores lasso: condiciones de optimalidad, solución cerrada por umbral suave, descenso por coordenadas y el algoritmo LARS |
| 4 | Árboles de decisión | Revisión de la solución del Ejercicio 6 (sección 12): dónde está el error y cuál es el árbol correcto |
| 5 | AdaBoost | Derivación de $a_i=\tfrac12\ln\frac{1-P_i}{P_i}$, de la regla de pesos y de la cota exponencial del error de entrenamiento |

```{note}
Al revisar las notas encontré algunos detalles que corrijo en el capítulo correspondiente: una cadena de desigualdades
que no se cumple en el capítulo de K-NN, el índice de la suma en la función objetivo de lasso, los paréntesis del
paso 3 del algoritmo AdaBoost y varios conteos en la solución del Ejercicio 6.
```
