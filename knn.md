# K-NN: riesgo verdadero y error de la regla 1-NN

## Las tres formas del riesgo verdadero (ecuación 2.2)

La Definición 2.3 de las notas escribe el riesgo verdadero de una regla $h:\mathcal X\to\mathcal Y$ de tres maneras:

$$
L_{\mathcal D}(h):=\underset{(\boldsymbol x,y)\sim\mathcal D}{\mathbb P}[h(\boldsymbol x)\neq y]
:=\mathcal D\big(\{(\boldsymbol x,y):\ h(\boldsymbol x)\neq y\}\big).
$$ (eq-riesgo)

Hay que demostrar que esas expresiones están bien definidas y son iguales, y que el riesgo empírico $L_S(h)$ es un
buen sustituto de $L_{\mathcal D}(h)$ cuando no conocemos $\mathcal D$.

**Planteamiento.** Sea $(\Omega,\mathcal F,\mathbb P)$ un espacio de probabilidad y $(\boldsymbol x,y):\Omega\to\mathcal X\times\mathcal Y$ un
vector aleatorio con ley $\mathcal D$, es decir, $\mathcal D(B)=\mathbb P\big((\boldsymbol x,y)\in B\big)$ para todo conjunto medible
$B\subseteq\mathcal X\times\mathcal Y$. En clasificación $\mathcal Y$ es finito (por ejemplo $\{0,1\}$) y $h$ es medible. Llamamos
$A:=\{(\boldsymbol x,y):\ h(\boldsymbol x)\neq y\}$ al conjunto de errores.

````{prf:proposition} El conjunto de errores es medible
:label: prop-medible
$A$ es un conjunto medible de $\mathcal X\times\mathcal Y$, así que $\mathcal D(A)$ está bien definido.
````

```{prf:proof}
Como $\mathcal Y$ es finito, $h$ se equivoca en $(\boldsymbol x,y)$ exactamente cuando predice alguna clase $c$ y la
etiqueta verdadera es otra:

$$
A=\bigcup_{c\in\mathcal Y}\Big(h^{-1}(\{c\})\times(\mathcal Y\setminus\{c\})\Big).
$$

Cada $h^{-1}(\{c\})$ es medible porque $h$ es medible, y $\mathcal Y\setminus\{c\}$ es medible porque $\mathcal Y$ es finito. Un
producto de conjuntos medibles es medible y una unión finita de medibles es medible, así que $A$ es medible.
```

````{prf:proposition} Igualdad de las tres expresiones
:label: prop-tres-formas
$$
\underset{(\boldsymbol x,y)\sim\mathcal D}{\mathbb P}[h(\boldsymbol x)\neq y]=\mathcal D(A)=\mathbb E_{(\boldsymbol x,y)\sim\mathcal D}\big[\mathbb 1\{h(\boldsymbol x)\neq y\}\big].
$$
````

```{prf:proof}
*Primera igualdad.* El evento $\{h(\boldsymbol x)\neq y\}$ es el mismo que $\{(\boldsymbol x,y)\in A\}$. Por la definición de ley,

$$
\mathbb P[h(\boldsymbol x)\neq y]=\mathbb P\big((\boldsymbol x,y)\in A\big)=\mathcal D(A).
$$

*Segunda igualdad.* La función indicadora $\mathbb 1_A$ es simple: vale $1$ en $A$ y $0$ fuera de $A$. La integral de una
función simple es la suma de sus valores por la medida de cada pieza:

$$
\mathbb E\big[\mathbb 1\{h(\boldsymbol x)\neq y\}\big]=\int_{\mathcal X\times\mathcal Y}\mathbb 1_A\,d\mathcal D=1\cdot\mathcal D(A)+0\cdot\mathcal D(A^c)=\mathcal D(A).
$$
```

Esta última forma es la que usa la demostración del Lema 2.1: $L_{\mathcal D}(h)$ es la **pérdida 0-1 esperada**.

````{prf:proposition} El riesgo empírico estima sin sesgo al riesgo verdadero
:label: prop-insesgado
Si $S=((\boldsymbol x_1,y_1),\dots,(\boldsymbol x_m,y_m))\sim\mathcal D^m$ es una muestra i.i.d. y $h$ no depende de $S$, entonces

$$
\mathbb E_{S}[L_S(h)]=L_{\mathcal D}(h),\qquad \operatorname{Var}_S[L_S(h)]=\frac{L_{\mathcal D}(h)\big(1-L_{\mathcal D}(h)\big)}{m}\le\frac{1}{4m}.
$$
````

```{prf:proof}
Sea $Z_i:=\mathbb 1\{h(\boldsymbol x_i)\neq y_i\}$. Entonces $L_S(h)=\frac1m\sum_{i=1}^m Z_i$, porque contar los índices con error y
dividir por $m$ es lo mismo que promediar las indicadoras. Por la {prf:ref}`prop-tres-formas`, cada $Z_i$ es una Bernoulli
con parámetro $L:=L_{\mathcal D}(h)$, así que $\mathbb E[Z_i]=L$ y $\operatorname{Var}(Z_i)=L(1-L)$.

Por linealidad de la esperanza, $\mathbb E[L_S(h)]=\frac1m\sum_i L=L$. Como los $Z_i$ son independientes, la varianza de la
suma es la suma de las varianzas: $\operatorname{Var}[L_S(h)]=\frac1{m^2}\,m\,L(1-L)=\frac{L(1-L)}m$. Por último,
$L(1-L)\le\frac14$ porque $L(1-L)=\frac14-(L-\frac12)^2$.
```

Por la ley de los grandes números, $L_S(h)\to L_{\mathcal D}(h)$ cuando $m\to\infty$, y la desigualdad de Hoeffding da una
velocidad: $\mathbb P\big(|L_S(h)-L_{\mathcal D}(h)|>\varepsilon\big)\le 2e^{-2m\varepsilon^2}$. Esto justifica minimizar $L_S$ (ERM)
cuando no conocemos $\mathcal D$. La hipótesis "*$h$ no depende de $S$*" es importante: si $h$ se eligió mirando $S$, el
riesgo de entrenamiento subestima al verdadero.

````{prf:proposition} Riesgo en términos de $\eta$ y error de Bayes
:label: prop-bayes
Sea $\mathcal Y=\{0,1\}$ y $\eta(\boldsymbol x):=\mathbb P[y=1\mid\boldsymbol x]$. Para toda regla $h$,

$$
L_{\mathcal D}(h)=\mathbb E_{\boldsymbol x}\Big[\eta(\boldsymbol x)\,\mathbb 1\{h(\boldsymbol x)=0\}+\big(1-\eta(\boldsymbol x)\big)\,\mathbb 1\{h(\boldsymbol x)=1\}\Big],
$$

y la regla de Bayes $h^\star(\boldsymbol x)=\mathbb 1\{\eta(\boldsymbol x)\ge\tfrac12\}$ alcanza el mínimo

$$
L_{\mathcal D}(h^\star)=\mathbb E_{\boldsymbol x}\big[\min\{\eta(\boldsymbol x),1-\eta(\boldsymbol x)\}\big]\le L_{\mathcal D}(h).
$$ (eq-bayes)
````

```{prf:proof}
Por la ley de la esperanza total, $L_{\mathcal D}(h)=\mathbb E_{\boldsymbol x}\big[\mathbb P(h(\boldsymbol x)\neq y\mid\boldsymbol x)\big]$. Dado $\boldsymbol x$, la
predicción $h(\boldsymbol x)$ es fija: si vale $0$, hay error cuando $y=1$, con probabilidad $\eta(\boldsymbol x)$; si vale $1$, hay error
cuando $y=0$, con probabilidad $1-\eta(\boldsymbol x)$. Eso da la primera fórmula.

Para cada $\boldsymbol x$, el integrando es $\eta(\boldsymbol x)$ o $1-\eta(\boldsymbol x)$, según lo que prediga $h$, así que nunca es menor
que $\min\{\eta(\boldsymbol x),1-\eta(\boldsymbol x)\}$. La regla de Bayes predice $1$ justo cuando $1-\eta\le\eta$, es decir, siempre elige
el menor de los dos, y su integrando es exactamente ese mínimo. Integrando en $\boldsymbol x$ se obtiene {eq}`eq-bayes`.
```

## La desigualdad $2\eta(1-\eta)\le 2\min\{\eta,1-\eta\}$

Al final del Lema 2.1 se usa que

$$
\mathbb E_{\boldsymbol x}\big[2\eta(\boldsymbol x)(1-\eta(\boldsymbol x))\big]\le 2\,\mathbb E_{\boldsymbol x}\big[\min\{\eta(\boldsymbol x),1-\eta(\boldsymbol x)\}\big]=2L_{\mathcal D}(h^\star).
$$ (eq-lema-final)

Las notas tratan el caso $\min\{\eta,1-\eta\}=\eta$; aquí falta el caso $\min\{\eta,1-\eta\}=1-\eta$. Escribo $t:=\eta(\boldsymbol x)\in[0,1]$.

````{prf:proposition} Caso $\min\{t,1-t\}=1-t$
:label: prop-caso
Si $t\in[0,1]$ y $\min\{t,1-t\}=1-t$, entonces $2t(1-t)\le 2\min\{t,1-t\}$.
````

```{prf:proof}
Que el mínimo sea $1-t$ significa que $1-t\le t$, es decir, $t\ge\tfrac12$. Además $1-t\ge0$ y $t\le1$. Multiplicando
la desigualdad $t\le 1$ por el número no negativo $2(1-t)$, el sentido se conserva:

$$
2t(1-t)\le 2\cdot 1\cdot(1-t)=2(1-t)=2\min\{t,1-t\}.
$$
```

Los dos casos se pueden escribir en una sola línea, porque el producto de dos números es su mínimo por su máximo:

$$
2t(1-t)=2\,\min\{t,1-t\}\cdot\max\{t,1-t\}\le 2\min\{t,1-t\},
$$

ya que $\max\{t,1-t\}\le 1$. Como la desigualdad vale para cada $\boldsymbol x$, también vale al tomar esperanza (la esperanza
conserva el orden), y con la {prf:ref}`prop-bayes` se obtiene {eq}`eq-lema-final`. Así se completa el Lema 2.1:

$$
\underset{S\sim\mathcal D^m}{\mathbb E}[L_{\mathcal D}(h_S)]\le 2L_{\mathcal D}(h^\star)+c\,\underset{S\sim\mathcal D^m,\ \boldsymbol x\sim\mathcal D_x}{\mathbb E}\big[\|\boldsymbol x-\boldsymbol x_{\pi_1(\boldsymbol x)}\|\big].
$$

```{warning} Observación sobre la cadena de las notas
En el caso $\min\{t,1-t\}=t$ las notas escriben $2t(1-t)\le 2(1-t)^2\le 2\min\{t,1-t\}$. La primera desigualdad es
cierta ($t\le 1-t$), pero la segunda no: con $t=0.1$ queda $2(0.9)^2=1.62$, que no es menor que $2\min\{0.1,0.9\}=0.2$.
La versión correcta es la misma idea de arriba: $2t(1-t)\le 2t\cdot 1=2\min\{t,1-t\}$, porque $1-t\le1$.

Por la misma razón, en el caso $\min=1-t$ **no** sirve la cadena simétrica $2t(1-t)\le 2t^2\le 2(1-t)$: con $t=0.9$,
$2t^2=1.62>0.2=2(1-t)$.
```

**Comprobación numérica.** Con una rejilla de $10\,001$ valores de $t$ en $[0,1]$, el máximo de
$2t(1-t)-2\min\{t,1-t\}$ es $0$ y solo se alcanza en $t=0$ y $t=1$, así que la desigualdad nunca se viola. En
cambio, $2(1-t)^2\le 2t$ equivale a $t^2-3t+1\le0$, que solo se cumple para $t\ge\frac{3-\sqrt5}{2}\approx0.382$: en
todo el tramo $0\le t<0.382$ la cadena de las notas falla, y en la rejilla la diferencia llega a $2$ en $t=0$.
