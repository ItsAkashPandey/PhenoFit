# PhenoFit Pro

**PhenoFit Pro** is a Python-based interactive GUI tool for fitting **double logistic curves** to vegetation index time-series data (such as GCC, NDVI, EVI, ExG, RGBVI).  
It is designed specifically for crop phenology research using PhenoCam, UAV, or satellite data.

---

## 📋 Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Step-by-Step Usage Guide](#step-by-step-usage-guide)
- [Screenshots](#screenshots)
- [Output Files](#output-files)
- [License](#license)
- [Author & Contact](#author--contact)

---

## 🌟 Features

- Double logistic curve fitting with GUI
- Slider and spin box control of parameters
- Smart initial parameter estimation
- SOS, EOS, and Peak annotations
- Grouping visualization for crop stages
- Constrained optimization with fallback logic
- Export of fitted values and plots

---

## 🧰 Requirements

- Python 3.7 or higher
- Packages:
  - pyqt5
  - matplotlib
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - openpyxl

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 🛠 Installation

### 1. Install Python

Go to https://www.python.org/downloads/ and install Python 3.7+

Make sure to check **“Add Python to PATH”** during installation.

---

### 2. Clone the Repository

```bash
git clone https://github.com/yourusername/PhenoFit-Pro.git
cd PhenoFit-Pro
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you face any issues, try using a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate    # On Windows
# or
source venv/bin/activate   # On Linux/Mac

pip install -r requirements.txt
```

---

## 🚀 Getting Started

### Run the Application

```bash
python PhenoFit_Pro.py
```

The GUI will open.

---

## 📚 Step-by-Step Usage Guide

### 1. Load Main Data

- Click **Load Data File**
- Select your `.csv` or `.xlsx` file with columns like `DAS`, `GCC`, `NDVI`, etc.
- Choose X (time axis) and Y (index) columns

### 2. (Optional) Load Grouping Data

- Click **Load Grouping Data**
- Load an Excel file with:
  - Start column (required)
  - End column (optional)
  - Label (stage names like "Tillering", "Heading")
  - Color (optional)

### 3. Adjust Parameters

You can:
- Use sliders for rough tuning
- Use spin boxes for exact values
- Lock any parameter using checkboxes

### 4. Optimize Curve

- Click **Optimize Fit**
- The app will try fitting the best double logistic curve
- Ensures biologically meaningful constraints like SOS < EOS

### 5. Visualize SOS, EOS, Peak

Check or uncheck the box: **"Show SOS/EOS/Peak"** to toggle markers.

### 6. Export Result

- Click **Download Graph + Excel**
- This will save:
  - `.xlsx` file with observed + fitted data
  - `.png` high-res image of the curve

---

## 🖼 Screenshots

### Load Main Data
![Load Data Screenshot](screenshots/load_data.png)

### Load Grouping
![Grouping Screenshot](screenshots/load_grouping.png)

### Fitting Curve + Stats
![Fitting Screenshot](screenshots/curve_fit.png)

> Upload your screenshots in a `screenshots/` folder in the repo

---

## 📂 Output Files

- **Excel Output** (`.xlsx`):
  - Sheet 1: Observed + Fitted values
  - Sheet 2: Parameters used
  - Sheet 3: Grouping stages (if provided)
- **PNG Output** (`.png`):
  - High-quality plot image

---

## 📜 License

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software...

(Include full MIT text in `LICENSE` file)

---

## 👤 Author & Contact

**Akash Kumar (aka Akash Pandey)**  
PhD Scholar, Geomatics Engineering, IIT Roorkee  
Research Focus: Remote Sensing, Crop Phenology, PhenoCam, AI  
Email: akash_k@ce.iitr.ac.in  

---

## 📣 Feedback & Issues

If you find any bug or want a new feature, open an issue at:

👉 https://github.com/yourusername/PhenoFit-Pro/issues

---

## 🙏 Acknowledgments

This GUI was developed under the PhD research project titled:  
**"Wheat Growth Sensitivity to Microclimate: A PhenoCam-AWS Fusion Approach"**