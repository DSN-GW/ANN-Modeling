
# Detailed Summary of: Neurocognitive Mechanisms of Social Inferences in Typical and Autistic Adolescents

**Authors**: Gabriela Rosenblau, Christoph W. Korn, Abigail Dutton, Daeyeol Lee, Kevin A. Pelphrey  
**Published In**: *Biological Psychiatry: Cognitive Neuroscience and Neuroimaging* (2021)  
**DOI**: [10.1016/j.bpsc.2020.07.002](https://doi.org/10.1016/j.bpsc.2020.07.002)

---

## 1. Background and Motivation

Social learning—inferring what others like or think—is crucial during adolescence. Individuals with Autism Spectrum Disorder (ASD) often struggle with social cognition. This paper seeks to understand the **mechanistic and neural basis** of these differences using **computational modeling** and **fMRI**.

Key concepts:
- **Prediction Error (PE)**: Difference between expected and actual feedback.
- **Preference Structures**: How preferences are organized in relation to one another.

---

## 2. Experimental Design

### Participants
- **Online Sample**: 99 adolescents for population preference structure.
- **fMRI Sample**: 26 TD (Typically Developing) adolescents and 20 ASD adolescents.

### Task Description
Participants were shown items (e.g., cookies, apples) and asked to infer how much a peer liked them. They received feedback about the actual rating of the peer and could use that to improve predictions.


---

## 3. Computational Models

### Model Overview

| Model Name           | Description |
|----------------------|-------------|
| No-Learning          | Uses own preferences only |
| RL-Ratings           | Rescorla-Wagner PE updating |
| Combination (Comb)   | Mix of RL and own preferences |
| Simple Prior         | Uses population mean preferences |
| Comb Simple Prior    | RL + population preferences |
| Similarity-RL        | RL scaled by preference similarity |
| Similarity Comb      | RL + population preferences + similarity scaling |



---

## 4. Results Summary

### Behavioral Results
- **TD Adolescents**: Better generalization and use of category information.
- **ASD Adolescents**: Descriptions lacked abstraction; relied on own preferences.

### Model Comparison

| Group | Best Model        | Description |
|-------|-------------------|-------------|
| TD    | Similarity Comb   | Integrated peer knowledge and similarity-weighted PE updating |
| ASD   | No-Learning / Comb | Minimal updating, heavy reliance on own preferences |



---

## 5. Neuroimaging Findings

### fMRI Results

| Brain Region        | TD Group Activity        | ASD Group Activity       |
|---------------------|--------------------------|---------------------------|
| Putamen/Caudate     | Encoded model-based PEs  | —                         |
| MPFC                | Model-free PE correlation| —                         |
| Angular Gyrus       | —                         | Encoded own preferences   |



---

## 6. Implications

- TD adolescents build **abstract social representations** and flexibly update beliefs.
- ASD adolescents use **egocentric** learning strategies and show less feedback adaptation.
- The modeling approach is critical for understanding individual social learning strategies and their neural basis.

---

## 7. Conclusion

This work combines **behavioral**, **computational**, and **neuroimaging** approaches to reveal:
- How social inferences differ in ASD.
- That effective social learning in TD relies on population knowledge and similarity metrics.
- ASD-related deficits stem from reduced learning from social feedback and reliance on self-reference.

---

## References
Full references available in the original article.

---

