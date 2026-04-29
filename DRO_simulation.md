Great. For **Simulation 1A**, I would make it a **single-observed-environment, latent-mixture training problem** whose purpose is very narrow:

> Can DRO identify the stable-and-predictive variable $X_2$ more often than ERM when $X_3$ looks strong in-sample only because of latent composition?

That is already enough to kick off the simulation section, and it maps very well to your paper’s framing that a super predictor combines **predictability** and **stability under plausible local shifts**.  

## 1. Simulation 1A: data-generating process

Use three core regressors:

* $X_1$: stable but not predictive
* $X_2$: stable and predictive
* $X_3$: predictive in the pooled training data, but unstable across latent mixture types

Let the latent mixture indicator be $U_i \in \{A,B\}$, unobserved by the researcher.

### Step 1: latent subtype

For each observation $i$,

$$
U_i \sim \text{Bernoulli}(\pi_{\text{train}})
$$

where $U_i=A$ with probability $\pi_{\text{train}}$, and $U_i=B$ otherwise.

For the baseline, set

$$
\pi_{\text{train}} = 0.8.
$$

So the training sample is mostly type $A$, which is what makes $X_3$ deceptively attractive.

### Step 2: predictors

Start simple:

$$
X_{i1}, X_{i2}, X_{i3} \stackrel{iid}{\sim} N(0,1),
$$

independent of each other and independent of $U_i$.

You can optionally add $p_{\text{noise}}$ extra noise predictors,

$$
X_{i4},\dots,X_{i,p} \stackrel{iid}{\sim} N(0,1),
$$

all irrelevant.

For the very first version, $p=3$ is fine. If you want a more realistic screening problem, use $p=10$ or $p=20$.

### Step 3: outcome equation

Define

$$
Y_i = \beta_2 X_{i2} + \beta_3(U_i) X_{i3} + \varepsilon_i,
$$

with

$$
\beta_1 = 0,\qquad \beta_2 = 1.
$$

Now make $X_3$ unstable through the latent mixture:

$$
\beta_3(A)=1.5,\qquad \beta_3(B)=-0.5.
$$

And let

$$
\varepsilon_i \sim N(0,\sigma^2).
$$

This gives exactly the taxonomy you wanted:

* $X_1$: stable, not predictive
* $X_2$: stable, predictive
* $X_3$: predictive in the pooled training sample, but its effect depends on latent composition

Because $\pi_{\text{train}}=0.8$, the pooled average effect of $X_3$ is

$$
E[\beta_3(U)] = 0.8(1.5)+0.2(-0.5)=1.1,
$$

so in the training sample $X_3$ will often look at least as strong as, and often stronger than, $X_2$. That is what makes it deceptive.

## 2. What the simulation is designed to show

This simulation does **not** yet require an external test set. Its main question is selection:

* Does ERM tend to rank or select $X_3$ because it looks strong in the observed sample?
* Does DRO more often favor $X_2$, because $X_2$ is less fragile to local perturbations of the empirical distribution?

That fits your paper’s logic exactly: ERM targets fit under the empirical distribution, while DRO minimizes worst-case loss over a Wasserstein ball around the empirical distribution. 

## 3. ERM benchmark

Use the linear LAD benchmark first, not OLS, so that the comparison is clean and directly aligned with Chen and Paschalidis.

The ERM-LAD problem is

$$
\hat\beta^{\text{ERM}}
\in
\arg\min_{\beta}
\frac{1}{n}\sum_{i=1}^n |Y_i - X_i'\beta|.
$$

This is just empirical risk minimization under absolute loss.

You can also report OLS later as an auxiliary benchmark, but for the first simulation I would compare:

* **ERM-LAD**
* **Wasserstein DRO-LAD**

That keeps the loss fixed and isolates the effect of robustness.

## 4. DRO training objective

Your working paper frames the robust learning problem generically as

$$
\hat\theta^{\text{DRO}}
\in
\arg\min_{\theta\in\Theta}
\sup_{Q\in U_W(\hat P_n;\epsilon)} E_Q[\ell(\theta;Z)],
$$

where the ambiguity set is a Wasserstein ball around the empirical distribution. 

For linear regression with absolute deviation loss, Chen and Paschalidis show that when you use the order-1 Wasserstein metric induced by a norm, the DRO problem becomes a regularized LAD problem. They emphasize that $t=1$ is the appropriate choice here; taking $t>1$ removes the useful Wasserstein-ball structure in this setup. 

The resulting estimator is

$$
\hat\beta^{\text{DRO}}
\in
\arg\min_{\beta}
\frac{1}{n}\sum_{i=1}^n |Y_i - X_i'\beta|
+
\epsilon \|(-\beta,1)\|_*,
$$

where $\|\cdot\|_*$ is the dual norm associated with the ground norm used in the Wasserstein metric. Chen and Paschalidis restate the linear-regression Wasserstein formulation in exactly this form in their experiments. 

That objective is perfect for your first simulation because it is transparent:

* first term = empirical fit
* second term = robustness penalty, with size controlled by $\epsilon$

This is also tightly connected to your paper’s predictability–stability decomposition: the robust loss is bounded by empirical loss plus a stability penalty proportional to $\epsilon$ times a Lipschitz constant.  

## 5. Which norm to use

For Simulation 1A, I would use **$W_1$ with an $\ell_2$ ground norm** on $(x,y)$. Then the dual norm is also $\ell_2$, and the DRO estimator becomes

$$
\hat\beta^{\text{DRO},2}
\in
\arg\min_{\beta}
\frac{1}{n}\sum_{i=1}^n |Y_i - X_i'\beta|
+
\epsilon \sqrt{\|\beta\|_2^2 + 1}.
$$

This is the clean default. Chen and Paschalidis compare $\ell_2$- and $\ell_\infty$-induced Wasserstein formulations and note that the right norm depends on the underlying structure: $\ell_2$ is natural for dense signals, while $\ell_\infty$ leads to an $\ell_1$-type penalty that is more aligned with sparse structure. 

Since your baseline DGP has only a few variables and is not primarily about sparsity, $\ell_2$ is the best starting point.

## 6. How to interpret the DRO objective in your simulation

In your paper’s language, the robust objective asks whether a predictor remains useful once you allow **adverse but plausible local perturbations** of the observed joint distribution.  

In Simulation 1A, this matters because $X_3$ is strong only because the training sample overrepresents latent type $A$. So:

* ERM-LAD tends to reward $X_3$'s in-sample strength
* DRO-LAD should be less willing to lean on $X_3$, because $X_3$'s contribution is more sensitive to local reweightings and perturbations of the empirical distribution

That is exactly the point you want the simulation to make.

## 7. Empirical selection of the radius $\epsilon$

Chen and Paschalidis discuss theory-based guidance for choosing the radius, including measure concentration ideas, but they also state plainly that **in practice cross-validation is usually adopted**, although it can be computationally expensive. They motivate the radius as the size of a Wasserstein ball large enough to include the true distribution with high confidence, i.e. one wants

$$
W_{s,1}(P^*,\hat P_n)\le \epsilon,
$$

but operationally they note that practitioners usually tune $\epsilon$ empirically. 

They also use cross-validated radius selection in their numerical work. In the multi-output experiments, they explicitly say the optimal Wasserstein radius is chosen through cross-validation. 

So for your simulation, the clean rule is:

### Main implementation rule

Select $\epsilon$ by validation over a grid.

### Recommended grid

After standardizing predictors and centering/scaling the response, use something like

$$
\epsilon \in
\{0,\ 10^{-4},\ 3\cdot 10^{-4},\ 10^{-3},\ 3\cdot 10^{-3},\ 10^{-2},\ 3\cdot 10^{-2},\ 10^{-1},\ 3\cdot 10^{-1}\}.
$$

Including $\epsilon=0$ is useful because it nests ERM-LAD.

## 8. How to do cross-validation in this simulation

For Simulation 1A, because you are still in the “single observed environment” stage, I would use a simple split inside the training sample:

* 60% train
* 20% validation
* 20% holdout summary set

or 5-fold CV if you prefer.

But there is an important nuance: since your paper is about latent composition instability, you may want the validation stage to mimic **mild local perturbation** rather than plain random resampling.

So I would recommend two implementations:

### Version A: standard CV

Use ordinary K-fold CV to pick $\epsilon$. This is closest to the practice described by Chen and Paschalidis.

### Version B: perturbation-aware validation

Within each Monte Carlo replication:

* estimate the training sample’s latent composition only implicitly through the data
* create validation subsamples by reweighting or resampling to induce mild composition variation
* choose $\epsilon$ that minimizes average validation LAD loss across those perturbed validation samples

Version B is more faithful to your project’s spirit, but Version A is easier to implement first.

For Simulation 1A, I would start with **Version A** and mention that stress-test tuning comes next.

## 9. Step-by-step simulation algorithm

For each Monte Carlo replication:

### Generate data

1. Draw $U_i \sim \text{Bernoulli}(0.8)$, $i=1,\dots,n$.
2. Draw $X_{ij} \sim N(0,1)$, independent across $i,j$.
3. Draw $\varepsilon_i \sim N(0,\sigma^2)$.
4. Construct

   ```math
   Y_i = 1\cdot X_{i2} + \beta_3(U_i) X_{i3} + \varepsilon_i,
   ```
   with $\beta_3(A)=1.5$, $\beta_3(B)=-0.5$.

### Preprocess

5. Standardize the predictors using the training sample.
6. Center $Y$, optionally scale it.

### Fit ERM-LAD

7. Solve

```math
\hat{\beta}^{\text{ERM}}
\in
\arg\min_{\beta}
\frac{1}{n}\sum_i |Y_i-X_i'\beta|
```

### Fit DRO-LAD

8. For each $\epsilon$ in the grid, solve

  ```math
\hat{\beta}^{\text{DRO}}(\epsilon)
\in
\arg\min_{\beta}
\frac{1}{n}\sum_i |Y_i-X_i'\beta|
+
\epsilon \sqrt{\|\beta\|_2^2+1}
```

9. Choose the best $\epsilon$ by CV.

### Compute variable-importance output

10. Record:

* estimated coefficients $\hat\beta_1,\hat\beta_2,\hat\beta_3$
* ranking of $|\hat\beta_j|$
* whether $X_2$ is ranked above $X_3$
* optionally your LOVO-style importance if you want to compute it

### Repeat

11. Repeat over many replications, say 500 or 1,000.

## 10. What to report for Simulation 1A

The core outcome should be **selection frequency**, not test MSE yet.

I would report:

* $P(\text{rank } X_2 > X_3)$
* $P(\text{top predictor is } X_2)$
* $P(\text{top predictor is } X_3)$
* mean and median estimated coefficients for $X_2$ and $X_3$

A clean table would be:

| Method                        | Pr(top = X2) | Pr(top = X3) | Mean rank of X2 | Mean rank of X3 |
| ----------------------------- | -----------: | -----------: | --------------: | --------------: |
| ERM-LAD                       |          ... |          ... |             ... |             ... |
| DRO-LAD $(\ell_2$, CV radius) |          ... |          ... |             ... |             ... |

The expected result is:

* ERM more often crowns $X_3$
* DRO more often crowns $X_2$

## 11. Suggested baseline parameter choices

For the first run:

* $n = 100$ or $150$
* $p = 3$ initially
* $\pi_{\text{train}}=0.8$
* $\beta_2=1$
* $\beta_3(A)=1.5$
* $\beta_3(B)=-0.5$
* $\sigma = 1$

These values make $X_3$ strong enough to deceive ERM, but not so overwhelming that $X_2$ disappears.

## 12. What you should say in the paper

A compact simulation description could read like this:

> We consider a single observed training environment with latent compositional heterogeneity. One predictor is stable but null, one is stable and predictive, and one is predictive only because the training sample overrepresents a latent subgroup in which its effect is strong. We estimate linear LAD and Wasserstein-DRO LAD models and ask which method more often identifies the stable-and-predictive variable as the top predictor. The DRO estimator follows the Chen and Paschalidis formulation, which reduces the Wasserstein robust LAD problem to empirical LAD plus a norm penalty on the extended coefficient vector, with the radius selected by cross-validation.  

## 13. My practical recommendation

To get moving quickly:

* use **LAD ERM vs DRO-LAD**
* use **$W_1$ with $\ell_2$ ground norm**
* tune $\epsilon$ by **cross-validation on a grid**
* focus Simulation 1A purely on **selection of $X_2$ vs $X_3$**

Then Simulation 1B can add the stress-test curve under changing mixture weights.
