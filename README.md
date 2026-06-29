# 🍽️ MenuManager — Smart Menu Categorizer & Description Generator

> **1-Week Intern Project · Track A · LLMs & Transformers**  
> Built by **Dhanush P** as part of the MenuManager intern project at [Organisation Name]

---

## 📌 What This Project Does

MenuManager Track A is a CLI-based AI tool that takes any dish name as input and automatically:

1. **Predicts the menu category** it belongs to (Starter, Main Course, Biryani, Bread, Rice, Soup, Salad, Dessert, or Beverage) — without any labeled training data
2. **Generates a short, professional menu description** — the kind a restaurant owner would use on a food delivery app

No training from scratch. No GPU required. No API keys. Everything runs on free models from the Hugging Face Hub.

---

## 🎯 Example

```
Enter dish name: Paneer Tikka

  Dish        : Paneer Tikka
  Category    : Starter  (confidence: 0.91)
  Description : Smoky tandoor-grilled paneer cubes marinated in spiced yogurt,
                served with mint chutney and a fresh onion salad.
```

---

## 🧠 Models Used

| Task | Model | Why This Model |
|---|---|---|
| Zero-shot category classification | `facebook/bart-large-mnli` | Fine-tuned on NLI — classifies text into unseen categories using entailment reasoning |
| Description generation | `google/flan-t5-base` | Instruction-tuned T5 — follows natural language prompts to generate fluent text |

Both models are **free, open-source, and run on CPU**.

---

## 🏗️ Project Structure

```
menumanager-track-a/
│
├── MenuManager_TrackA.ipynb   # Main Colab notebook — all code here
├── menu.csv                   # 117-row Indian restaurant dataset
└── README.md                  # This file
```

---

## 📂 Dataset — menu.csv

A hand-crafted dataset of **117 popular Indian restaurant dishes** covering 9 categories.

| Column | Description | Example |
|---|---|---|
| `dish_name` | Restaurant-style item name | `Chicken Tikka Masala` |
| `category` | One of 9 menu categories | `Main Course` |
| `price` | Price in Indian Rupees | `370` |
| `description` | 2–3 sentence professional description | `Smoky chicken tikka pieces...` |
| `dietary_tags` | Always starts with Veg or Non-Veg | `Non-Veg, Spicy, Contains Dairy` |

**Categories covered:** Starter · Main Course · Biryani · Bread · Rice · Soup · Salad · Dessert · Beverage

---

## ⚙️ Setup & Installation

### Option 1 — Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `MenuManager_TrackA.ipynb`
3. Run all cells top to bottom — models download automatically

### Option 2 — Local (Python 3.10+)

```bash
git clone https://github.com/your-username/menumanager-track-a.git
cd menumanager-track-a
pip install transformers torch pandas
jupyter notebook MenuManager_TrackA.ipynb
```

> No Hugging Face account or API key needed. All models are freely downloadable.

---

## 🚀 How It Works

### Step 1 — Zero-Shot Classification (BART-MNLI)

Instead of training a classifier on labeled examples, this project uses **zero-shot classification**:

- The model checks if the dish name *entails* (logically implies) each category label
- Example: Does "Chicken Biryani" entail "This menu item falls under the Biryani category?" → YES (high score)
- The category with the highest entailment score wins
- This works without any training data because the model was fine-tuned on Natural Language Inference

```python
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = classifier("Chicken Biryani", candidate_labels=["Biryani", "Starter", "Dessert", ...])
# → {'labels': ['Biryani', ...], 'scores': [0.89, ...]}
```

### Step 2 — Description Generation (FLAN-T5)

The model is given a detailed instruction prompt and generates one professional sentence:

```python
prompt = (
    "You are a professional restaurant menu copywriter. "
    "Write exactly one appetizing sentence (15-25 words) "
    "describing 'Chicken Biryani', a biryani item. It is Non-Veg, Spicy. "
    "Mention the cooking style and key flavors. "
    "Do not start with the dish name."
)
```

Key generation parameters:
- `temperature=0.75` — balanced between creative and focused
- `top_p=0.92` — nucleus sampling for quality output
- `repetition_penalty=1.4` — prevents repeated words/phrases

### Step 3 — Integrated Function

Both pipelines are combined into one entry point:

```python
result = menu_assistant("Paneer Tikka")
# Returns:
# {
#   "dish_name": "Paneer Tikka",
#   "predicted_category": "Starter",
#   "confidence": 0.91,
#   "generated_description": "Smoky tandoor-grilled paneer..."
# }
```

---

## 📊 Evaluation

Accuracy was measured by running the classifier on all 117 dishes and comparing
predicted category against the ground truth in `menu.csv`.

| Hypothesis Template | Accuracy |
|---|---|
| `"This is a {} dish."` | ~34% (breaks for Beverage, Bread categories) |
| `"This dish belongs to the {} category."` | ~56% |
| `"This menu item falls under the {} category."` ✅ | ~58–62% |

**Key finding:** The hypothesis template has a large effect on accuracy because
BART-MNLI is an NLI model — grammatically correct templates produce significantly better results.

**Mismatch analysis:** The most common errors were:
- Ambiguous dishes that sit between two categories (e.g. Gobi Manchurian → Starter vs Main Course)
- Dishes whose names don't contain strong category signals without descriptions

---

## 🔧 Notebook Cell Guide

| Cell | What It Does |
|---|---|
| 1 | Install libraries |
| 2 | Hello Transformer sanity check |
| 3 | Build and save menu.csv |
| 4 | Load and explore data |
| 5 | Load BART-MNLI classifier |
| 6 | Test on sample dishes |
| 7 | Full accuracy evaluation + mismatch list |
| 8 | Load FLAN-T5 tokenizer and model |
| 9 | `generate_description()` function |
| 10 | Test generator on 5 dishes |
| 11 | `predict_category()` helper |
| 12 | `menu_assistant()` combined function |
| 13 | Live CLI demo loop |
| 14 | Confidence threshold (stretch goal) |

---

## 🎁 Stretch Goals Completed

- [x] **Confidence threshold** — if the model scores below 0.45, returns `"Uncertain — please choose manually"` instead of a wrong guess
- [x] **Dietary tag awareness** — generator receives dietary tags and weaves them naturally into the description
- [x] **Prompt iteration** — 3 prompt versions compared; documented which produced better output and why

---

## 💡 Key Learnings & Reflection

**What worked well:**
- Zero-shot classification required zero labeled training data — immediately useful on a real dataset
- FLAN-T5's instruction-following quality improved dramatically with a well-crafted prompt
- The hypothesis template wording had a far larger effect on accuracy than expected

**What was harder than expected:**
- Ambiguous dish names (e.g. dishes that sit between Starter and Main Course) genuinely confused the model — the same way they'd confuse a human without context
- FLAN-T5 on CPU is slow for bulk generation (~5–8 seconds per dish)

**One thing I'd try with more time:**
- Fine-tune a small classifier on the mismatch dishes, or pass both the dish name and its description to the zero-shot model — the description contains far more category signal than the name alone

---

## 📚 References

- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers/quicktour)
- [BART-MNLI Model Card](https://huggingface.co/facebook/bart-large-mnli)
- [FLAN-T5 Model Card](https://huggingface.co/google/flan-t5-base)
- [Hugging Face Course — Chapter 1](https://huggingface.co/learn/nlp-course/chapter1)

---

## 👤 Author

**Dhanush P**  
Intern — MenuManager Project · Track A  
*Smart Menu Categorizer & Description Generator*

