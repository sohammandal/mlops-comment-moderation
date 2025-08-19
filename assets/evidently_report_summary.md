## Evidently Model Evaluation Summary

### Reference Set (Original Test Data)

| Class     | Precision | Recall | F1-score | Support |
| --------- | --------- | ------ | -------- | ------- |
| 0 (clean) | 0.9647    | 0.9661 | 0.9654   | 57,735  |
| 1 (toxic) | 0.6825    | 0.6732 | 0.6778   | 6,243   |

**Accuracy:** 0.9376
**Macro Avg F1:** 0.8216
**Weighted Avg F1:** 0.9374

---

### Changed Set (Augmented Test Data)

| Class     | Precision | Recall | F1-score | Support |
| --------- | --------- | ------ | -------- | ------- |
| 0 (clean) | 0.9463    | 0.9816 | 0.9636   | 57,735  |
| 1 (toxic) | 0.7403    | 0.4844 | 0.5856   | 6,243   |

**Accuracy:** 0.9331
**Macro Avg F1:** 0.7746
**Weighted Avg F1:** 0.9267

---

### Drift Summary

* **Weighted F1:** 0.9374 → 0.9267 (**Δ -0.0106**)
* **Macro F1:** 0.8216 → 0.7746 (**Δ -0.0470**)

📊 [Evidently HTML Report](evidently_text_moderation_ref_vs_changed.html)

