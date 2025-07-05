# 📘 Course Syllabus – Phase 1  
**Applied Machine Learning and LLM Security Foundations**  
**Term:** Summer 2025 (Phase 1: July 8 – August 3, 2025)  
**Format:** Self-paced Independent Study  
**Credits:** 2 (Informal, self-assessed)  
**Instructor:** You  
**Delivery Mode:** Asynchronous / Self-directed  

---

## 📖 Course Description  
This 4-week independent study introduces the fundamentals of applied machine learning and key concepts in AI/LLM security. Students will build classical models, simulate adversarial attacks, and investigate vulnerabilities such as label poisoning and prompt injection. The phase concludes with a hands-on midterm project.

---

## 🔧 Prerequisites  
- Intermediate Python (functions, data structures, NumPy, Pandas)  
- Familiarity with Git and the command line  
- Some knowledge of Docker and cloud tools is helpful, not required  

---

## 🎯 Course Objectives  
By completing this course, you will be able to:
- Implement and evaluate baseline ML models using scikit-learn  
- Simulate adversarial attacks like data poisoning and prompt injection  
- Explore the attack surface of modern LLMs  
- Log experiments and analysis in a reproducible format  
- Complete a midterm project applying both ML and security concepts  

---

## 📚 Core Materials & Tools  

| Resource | Link |
|---------|------|
| ML Zoomcamp Curriculum | https://github.com/DataTalksClub/machine-learning-zoomcamp |
| Robust Intelligence Blog | https://robustintelligence.com/blog |
| Prompt Injection Database | https://promptinject.com |
| UCI SMS Spam Dataset | https://archive.ics.uci.edu/dataset/228/sms+spam+collection |
| Kaggle Spam Dataset | https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset |
| HuggingFace Transformers | https://huggingface.co/transformers |
| Weights & Biases | https://wandb.ai |
| GitHub | https://github.com |

---

## 📅 Weekly Breakdown

### 📆 Week 1: ML Foundations + Attack Surface Awareness  
**Dates:** July 8 – July 14, 2025

**ML Concepts:**
- Linear Regression, Logistic Regression  
- Overfitting, bias-variance tradeoff  
- Precision, Recall, F1 Score  

**Resources:**
- [ML Zoomcamp: Week 1](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/cohorts/2023/01-linear-regression)  
- [Adversarial ML Primer – Robust Intelligence](https://robustintelligence.com/blog/ml-security/the-emerging-threat-of-adversarial-ml)  
- [Paper: “Adversarial Examples Are Not Bugs…”](https://arxiv.org/abs/1905.02175)

**Deliverables:**
- Train a baseline spam classifier using logistic regression  
- Track metrics using W&B or GitHub  
- Submit a short summary or notebook of results

---

### 📆 Week 2: Tree Models + Label Poisoning  
**Dates:** July 15 – July 21, 2025

**ML Concepts:**
- Decision Trees, Random Forests  
- Feature importance (Gini, SHAP)  
- Model interpretability  

**Resources:**
- [ML Zoomcamp: Week 2](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/cohorts/2023/02-experiment-tracking)  

**Security Focus:**
- Manual label flipping (data poisoning)  
- Measure and compare accuracy degradation  

**Deliverables:**
- Train models with clean vs. poisoned data  
- Compare results in visualizations (charts, confusion matrix)  
- Write a short notebook summary  

---

### 📆 Week 3: Feature Engineering + Prompt Injection Simulation  
**Dates:** July 22 – July 28, 2025

**ML Concepts:**
- TF-IDF, CountVectorizer  
- Pipelines in `scikit-learn`  
- ROC Curve, AUC  

**Resources:**
- [ML Zoomcamp: Week 3](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/cohorts/2023/03-data-preparation)  
- [Prompt Injection Database](https://promptinject.com)  

**Security Focus:**
- Add prompt injection-style text to spam/ham messages  
- Simulate model behavior degradation  

**Deliverables:**
- Retrain model on injected samples  
- Evaluate model on clean vs. noisy text  
- Write up results and injection effectiveness  

---

### 📆 Week 4: Infrastructure, Reproducibility + Midterm Project  
**Dates:** July 29 – August 3, 2025

**ML Concepts:**
- Logging, model evaluation  
- Git, DVC, Weights & Biases  
- *(Optional)* Docker + FastAPI deployment  

**Resources:**
- [ML Zoomcamp: Week 4](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/cohorts/2023/04-evaluation)

**Security Focus:**
- Summary of all attacks used  
- Explore additional concepts (model fingerprinting, hallucination risks)

---

## 🟥 Midterm Project: Spam Classifier with Prompt Injection  
**Due Date:** Sunday, August 3, 2025

**Project Requirements:**
- Train a spam classifier using either dataset  
- Apply prompt injection techniques  
- Compare clean vs adversarial results  
- Publish on GitHub or Colab  
- Include README or PDF with:
  - What did you learn?
  - What were the attack outcomes?
  - How can defenses be improved?

---

## 🧠 Weekly Reflection Template

Submit a short reflection each week (in Notion, Markdown, or journal):

- What did I learn this week?  
- What confused me or needs more practice?  
- What went well?  
- How will I improve next week?

---

## 📊 Self-Evaluation Criteria

| Component | Weight |
|----------|--------|
| Weekly deliverables | 25% |
| Midterm project | 40% |
| Reflections | 10% |
| Consistency & completion | 25% |

**Passing Threshold:** 70% or higher (self-assessed)

---
