# Neurocognitive Mechanisms of Social Inferences in Typical and Autistic Adolescents

## 🧠 Objective
To investigate how **typically developing (TD)** and **autistic (ASD)** adolescents represent and update social knowledge about others' preferences, and how these processes manifest in brain activity.

---

## 🧪 Experimental Setup
- Participants rated how much 3 peer profiles liked different items inside an fMRI scanner.
- Trial-by-trial feedback on actual preferences.
- Own preferences collected post-scan.
- Population preferences from a separate sample of 99 adolescents.

---

## ⚙️ Modeling Framework

Seven computational models were tested:

### Model 1: No-Learning


$$
\text{ER} = \beta_0 + \beta_1 \cdot \text{OP}
$$


### Model 2: RL-Ratings (Rescorla-Wagner)
$$
\text{ER}_{t+1} = \text{ER}_t + \alpha \cdot (F_t - \text{ER}_t)
$$

### Model 3: Combination (Comb)
$$
\text{ER}_{t+1} = \gamma (\text{ER}_t + \alpha \cdot PE_t) + (1 - \gamma) \cdot OP_{t+1}
$$

### Model 4: Simple Prior
$$
\text{ER} = \beta_0 + \beta_1 \cdot MP
$$

### Model 5: Comb with Simple Prior
$$
\text{ER}_{t+1} = \gamma (\text{ER}_t + \alpha \cdot PE_t) + (1 - \gamma) \cdot MP_{t+1}
$$

### Model 6: Similarity-RL
$$
\text{ER}_{t+1} = \text{ER}_t + \alpha \cdot PE_t \cdot r(i, I)
$$

### Model 7: Similarity-Comb (Winning Model for TD)
$$
\text{ER}_{t+1} = \gamma [\text{ER}_t + \alpha \cdot PE_t \cdot r(i, I)] + (1 - \gamma) \cdot MP_{t+1}
$$

---

## 🧠 Neural Findings

| Group | Brain Encoding |
|-------|----------------|
| TD Adolescents | Prediction Errors (Putamen, Caudate) |
| ASD Adolescents | Own Preferences (Angular Gyrus, Precuneus) |

---

## 🧩 Key Comparison

| Feature | TD Adolescents | ASD Adolescents |
|--------|----------------|----------------|
| Use of Social Knowledge | ✅ (Similarity & Population) | ❌ (Own Preference Only) |
| Learning Rate | High (α) | Minimal |
| Best Model | Similarity-Comb | No-Learning or Simple Comb |

---

## 📈 Implications

- Better model fit to **Similarity-Comb** → Higher **social responsiveness**.
- Computational and neuroimaging approaches provide biomarkers for **ASD social learning differences**.

