# VISPAC Edge Prototype - Datasets Guide

## Available Datasets

### 1. **Low Risk** (`low_risk`)
- **Source**: Kaggle - Biosensor Student Health Fitness Data
- **Patients**: 7 virtual patients (IDs 1-7)
- **Samples**: ~85 samples per patient (total 600)
- **Characteristics**:
  - HR: 60-100 bpm (mean ~79 bpm)
  - SpO2: 95-100% (mean ~97.5%)
  - Healthy students data

### 2. **High Risk** (`high_risk`)
- **Source**: PhysioNet BIDMC Database
- **Patients**: 53 real ICU patients
- **Samples**: ~480 samples per patient (total 25,361)
- **Characteristics**:
  - HR: 44-139 bpm (mean ~89 bpm)
  - SpO2: 83-100% (mean ~97%)
  - Real intensive care unit data

## How It Works

### Patient Mapping

**Low Risk:**
```
Simulation     Dataset
----------     -------
Patient 1   →  Student Group 1 (85 samples)
Patient 2   →  Student Group 2 (85 samples)
Patient 3   →  Student Group 3 (85 samples)
...
```

**High Risk:**
```
Simulation     Dataset
----------     -------
Patient 1   →  BIDMC Patient 1 (468 samples)
Patient 2   →  BIDMC Patient 2 (481 samples)
Patient 3   →  BIDMC Patient 3 (481 samples)
...
```

### Risk Classification

- All patients have risk calculated **dynamically** based on vital signs (NEWS2)
- Risk can vary during simulation as data changes
- Patients with high-risk data will naturally be classified as such

## Usage

### Method 1: Environment Variable

```bash
# Low Risk
export DATASET_TYPE=low_risk
python vispac_edge_prototype.py

# High Risk
export DATASET_TYPE=high_risk
python vispac_edge_prototype.py
```

### Method 2: Interactive Menu

```bash
python run_simulation.py
```

The menu shows:
1. 🟢 LOW RISK - Stable patients
2. 🔴 HIGH RISK - Critical patients
3. ℹ️ View dataset statistics
4. 🔧 Advanced settings

### Method 3: Docker Compose

Edit `compose.yaml`:
```yaml
environment:
  - DATASET_TYPE=high_risk  # or low_risk
```

Run:
```bash
docker-compose up
```

## Processed File Format

```csv
patient_id,hr,spo2
1,66.00,97.20
1,92.00,99.73
2,91.00,100.00
2,90.00,100.00
...
```

**Important**: 
- `patient_id`: Patient/student ID
- `hr`: Heart rate in bpm (with decimals)
- `spo2`: Oxygen saturation in % (with decimals)

## Statistics

### Low Risk - Virtual Patients

| ID | Samples | HR (bpm) | SpO2 (%) |
|----|----------|----------|----------|
| 1  | 85       | 77.8±11.2 | 97.4±1.5 |
| 2  | 85       | 78.3±11.8 | 97.7±1.4 |
| 3  | 85       | 78.3±11.4 | 97.7±1.4 |
| 4  | 85       | 77.3±11.6 | 97.6±1.4 |
| 5  | 85       | 79.3±11.2 | 97.3±1.5 |
| 6  | 85       | 79.2±12.2 | 97.4±1.5 |
| 7  | 90       | 80.6±11.6 | 97.6±1.4 |

### High Risk - BIDMC Patients (First 7)

| ID | Samples | HR (bpm) | SpO2 (%) | Profile |
|----|----------|----------|----------|--------|
| 1  | 468      | 91.3±1.0  | 96.9±0.4 | Stable |
| 2  | 481      | 91.1±1.4  | 100.0±0.1| High saturation |
| 3  | 481      | 76.6±2.1  | 95.8±0.6 | Mild bradycardia |
| 4  | 481      | 92.4±3.1  | 93.0±0.2 | Mild hypoxemia |
| 5  | 475      | 98.2±0.8  | 99.4±0.6 | Mild tachycardia |
| 6  | 481      | 81.8±1.2  | 99.0±0.0 | Normal |
| 7  | 481      | 90.1±0.3  | 95.0±0.0 | Mild hypoxemia |


## 🔗 References

- **Kaggle Dataset**: https://www.kaggle.com/datasets/ziya07/biosensor-studenthealthfitnessdata
- **BIDMC Database**: https://physionet.org/content/bidmc/1.0.0/