"""ViSPAC Edge Layer - Patient Vital Signs Simulation and Compression.

This module implements the edge tier of the ViSPAC (Vital Signs Processing
and Compression) edge-fog-cloud architecture for healthcare IoT systems.

The edge layer is responsible for:
    - Simulating/collecting patient vital signs (HR, SpO2)
    - Applying lossy compression (SDT - Swinging Door Trending)
    - Applying lossless compression for network transmission (Huffman/LZW)
    - Batching data by risk level for efficient transmission
    - Adaptive sampling rate based on NEWS2 feedback from fog
    - Backoff algorithm to reduce transmission during stable periods

Key Algorithms:
    1. Swinging Door Trending (SDT):
       Lossy compression preserving trend information while reducing data points.
       Configured per-risk level (higher risk = more frequent sampling).
    
    2. Adaptive Sampling (Algorithm 1):
       Adjusts collection intervals based on patient risk level:
       - HIGH: 30s HR, 30s SpO2
       - MODERATE: 120s HR, 180s SpO2
       - LOW: 300s HR, 600s SpO2
       - MINIMAL: 600s HR, 900s SpO2
    
    3. Backoff Algorithm (Algorithm 3):
       When signals are stable for K consecutive samples (K_STABLE=3),
       doubles the collection interval up to a maximum (ic_max).
       Reduces network traffic during stable monitoring periods.
    
    4. Risk-based Batching:
       Groups samples by risk level with different timeouts and size limits:
       - HIGH: 15s timeout, 5KB limit (urgent transmission)
       - MODERATE: 60s timeout, 20KB limit
       - LOW: 150s timeout, 50KB limit
       - MINIMAL: 300s timeout, 50KB limit

Quality Metrics:
    PRD (Percent Root-mean-square Difference) is calculated for each
    compressed signal to quantify signal fidelity after lossy compression.

Communication:
    Supports both HTTP POST and MQTT for sending batches to the fog layer.
    The fog returns NEWS2 scores which trigger risk level updates.

Environment Variables:
    API_URL: Fog layer endpoint (default: http://127.0.0.1:8000/vispac/upload_batch)
    EDGE_ID: Unique identifier for this edge device
    DATASET_TYPE: 'low_risk' or 'high_risk'
    PATIENT_RANGE: Selection of patients to simulate (e.g., '1-10', 'all')
    EDGE_USE_MQTT: '1' to enable MQTT, '0' for HTTP only
    MQTT_BROKER: MQTT broker address (default: 127.0.0.1)
    MQTT_PORT: MQTT broker port (default: 1883)

Author: Mateus Roveda
Master's Dissertation - ViSPAC Project
"""

import time
import json
import random
import requests
import logging
import os
import uuid
import pandas as pd
import numpy as np
from compressors import SwingingDoorCompressor, Huffman, LZW

# ---------------- Logging Setup (must be first) ----------------
LOG_PATH = "logs/edge_log.txt"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, 'w'), logging.StreamHandler()])
log = logging.getLogger("vispac-edge")

# Try to import YAML for configuration
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# ---------------- Configuration ----------------
CONFIG_PATH = os.environ.get("CONFIG_PATH", "config/simulation.yaml")

def load_config():
    """Load simulation configuration from YAML file."""
    config = {
        "defaults": {"spo2_scale": 1, "on_o2": False, "consciousness": "A"},
        "datasets": {
            "high_risk": {"probabilities": {"on_o2": 0.4, "spo2_scale_2": 0.3, "altered_consciousness": 0.2},
                         "consciousness_options": ["V", "P"]},
            "low_risk": {"probabilities": {"on_o2": 0.1, "spo2_scale_2": 0.05, "altered_consciousness": 0.02},
                        "consciousness_options": ["V"]}
        },
        "patient_overrides": {},
        "vitals_ranges": {
            "high_risk": {"sys_bp": [85, 95], "temp_high": [38.8, 39.8], "temp_low": [35.0, 35.8], 
                         "rr_high": [25, 29], "rr_low": [8, 10]},
            "moderate_risk": {"sys_bp": [95, 108], "temp": [38.1, 39.0], "rr": [21, 24]}
        }
    }
    
    if YAML_AVAILABLE and os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    # Deep merge with defaults
                    for key in file_config:
                        if key in config and isinstance(config[key], dict):
                            config[key].update(file_config[key])
                        else:
                            config[key] = file_config[key]
            log.info(f"Configuration loaded from {CONFIG_PATH}")
        except Exception as e:
            log.warning(f"Failed to load config from {CONFIG_PATH}: {e}. Using defaults.")
    elif not YAML_AVAILABLE:
        log.warning("PyYAML not installed. Using default configuration. Install with: pip install pyyaml")
    
    return config

CONFIG = load_config()

# ---------------- Constants ----------------
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/vispac/upload_batch")

# Edge Identification
EDGE_ID = os.environ.get("EDGE_ID", f"edge-{uuid.uuid4().hex[:8]}")

# Dataset Configuration
DATASET_TYPE = os.environ.get("DATASET_TYPE", "low_risk")  # low_risk or high_risk
DATASET_PATHS = {
    "low_risk": "datasets/low_risk/low_risk_processed.csv",
    "high_risk": "datasets/high_risk/high_risk_processed.csv",
    "default": "dataset.csv"  # Legacy dataset for compatibility
}
DATASET_PATH = DATASET_PATHS.get(DATASET_TYPE, DATASET_PATHS["default"])

# Patient Range: "1-10" means patients 1 to 10, "5,10,15" means specific patients
# Empty or "all" means all available patients
PATIENT_RANGE = os.environ.get("PATIENT_RANGE", "all")

# Number of patients to simulate (0 = all available) - DEPRECATED, use PATIENT_RANGE
NUM_PATIENTS = int(os.environ.get("NUM_PATIENTS", "0"))
K_STABLE = 3
KEEP_ALIVE = 600
HUFF_MIN, LZW_MIN = 1 * 1024, 32 * 1024

ASSEMBLER_CONFIG = {
    "HIGH":     {"timeout": 15,  "size_limit": 5 * 1024},
    "MODERATE": {"timeout": 60,  "size_limit": 20 * 1024},
    "LOW":      {"timeout": 150, "size_limit": 50 * 1024},
    "MINIMAL":  {"timeout": 300, "size_limit": 50 * 1024},
}

# Latency metrics storage
LATENCY_METRICS = {
    "send_count": 0,
    "total_latency_ms": 0,
    "min_latency_ms": float('inf'),
    "max_latency_ms": 0,
    "by_risk": {"HIGH": [], "MODERATE": [], "LOW": [], "MINIMAL": []}
}

def parse_patient_range(range_str, available_ids):
    """
    Parse PATIENT_RANGE environment variable.
    Supports:
    - "all" or empty: all available patients
    - "1-10": range from index 1 to 10 (1-based, inclusive)
    - "5,10,15,20": specific patient IDs
    - "0:5" or "10:20": slice notation (0-based)
    """
    if not range_str or range_str.lower() == "all":
        return available_ids
    
    range_str = range_str.strip()
    
    # Slice notation: "0:5" means first 5 patients
    if ":" in range_str:
        parts = range_str.split(":")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else len(available_ids)
        return available_ids[start:end]
    
    # Range notation: "1-10" means patients at indices 1 to 10 (1-based, inclusive)
    if "-" in range_str and range_str.count("-") == 1:
        parts = range_str.split("-")
        try:
            start = int(parts[0]) - 1  # Convert to 0-based
            end = int(parts[1])  # Inclusive, so no -1
            return available_ids[max(0, start):min(end, len(available_ids))]
        except ValueError:
            pass
    
    # Comma-separated IDs: "5,10,15,20"
    if "," in range_str:
        requested_ids = [x.strip() for x in range_str.split(",")]
        # Try to match by ID value
        str_available = [str(x) for x in available_ids]
        matched = [available_ids[str_available.index(rid)] for rid in requested_ids if rid in str_available]
        return matched if matched else available_ids
    
    # Single patient ID
    try:
        idx = int(range_str) - 1  # Assume 1-based index
        if 0 <= idx < len(available_ids):
            return [available_ids[idx]]
    except ValueError:
        # Try to match by ID string
        str_available = [str(x) for x in available_ids]
        if range_str in str_available:
            return [available_ids[str_available.index(range_str)]]
    
    log.warning(f"Could not parse PATIENT_RANGE '{range_str}', using all patients")
    return available_ids

def log_latency_summary():
    """Log summary of latency metrics."""
    m = LATENCY_METRICS
    if m["send_count"] == 0:
        return
    avg = m["total_latency_ms"] / m["send_count"]
    log.info(f"[LATENCY SUMMARY] Edge {EDGE_ID}")
    log.info(f"  Total sends: {m['send_count']}")
    log.info(f"  Avg latency: {avg:.1f}ms | Min: {m['min_latency_ms']:.1f}ms | Max: {m['max_latency_ms']:.1f}ms")
    for risk, latencies in m["by_risk"].items():
        if latencies:
            avg_risk = sum(latencies) / len(latencies)
            log.info(f"  {risk}: {len(latencies)} sends, avg {avg_risk:.1f}ms")

def reconstruct_signal(original_signal_buf, compressed_signal):
    """Reconstruct a signal from SDT-compressed points using linear interpolation.
    
    After SDT compression, only significant points are retained. This function
    reconstructs the original signal length by interpolating between the
    compressed points, allowing PRD calculation.
    
    Args:
        original_signal_buf: List of (timestamp, value) tuples from original signal.
        compressed_signal: List of (timestamp, value) tuples from SDT output.
    
    Returns:
        numpy.ndarray: Interpolated values at original timestamps.
    """
    original_timestamps = [p[0] for p in original_signal_buf]
    compressed_timestamps = [p[0] for p in compressed_signal]
    compressed_values = [p[1] for p in compressed_signal]
    
    # Use NumPy's interpolation function
    reconstructed_values = np.interp(original_timestamps, compressed_timestamps, compressed_values)
    return reconstructed_values

def calculate_prd(original_signal, reconstructed_signal):
    """Calculate Percent Root-mean-square Difference (PRD) between signals.
    
    PRD is a standard metric for evaluating lossy compression quality in
    biomedical signals. Lower values indicate better fidelity.
    
    Formula:
        PRD = sqrt(sum((x_orig - x_recon)^2) / sum(x_orig^2)) * 100
    
    Interpretation:
        - PRD < 1%: Excellent (suitable for diagnosis)
        - PRD < 5%: Good (suitable for monitoring)
        - PRD < 10%: Acceptable (suitable for trend analysis)
        - PRD > 10%: Poor (may lose clinical significance)
    
    Args:
        original_signal: List or array of original signal values.
        reconstructed_signal: List or array of reconstructed signal values.
    
    Returns:
        float: PRD percentage value.

    """
    original = np.array(original_signal)
    reconstructed = np.array(reconstructed_signal)

    # Avoid division by zero if original signal is null
    sum_sq_original = np.sum(original**2)
    if sum_sq_original == 0:
        return 0.0

    sum_sq_diff = np.sum((original - reconstructed)**2)
    prd = np.sqrt(sum_sq_diff / sum_sq_original) * 100
    return prd
# -----------------------------------------------------------------


# --------------- Simulated Dataset --------------------
SAMPLE_DATA=[(80,98),(81,98),(80,99),(82,98),(83,98),(82,97),(85,98),(84,98),(86,99),
    (88,98),(90,97),(92,96),(95,95),(98,94),(105,93),(110,92),(112,91),(115,90),
    (118,91),(120,92),(115,93),(110,94),(105,95),(100,96),(95,97),(92,98),(90,98),
    (88,99),(86,98),(85,98)]

def ensure_dataset():
    """
    Loads the dataset based on DATASET_TYPE configuration.
    Raises an error if the dataset doesn't exist.
    """
    if not os.path.exists(DATASET_PATH):
        log.error(f"❌ Dataset not found: {DATASET_PATH}")
        log.error(f"")
        log.error(f"To download the datasets, run:")
        log.error(f"  - Low risk: Download from Kaggle and process")
        log.error(f"  - High risk: python download_bidmc_data.py")
        log.error(f"")
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    log.info(f"Dataset loaded: {DATASET_PATH}")
    log.info(f"  Type: {DATASET_TYPE}")
    log.info(f"  Samples: {len(df)}")
    log.info(f"  Average HR: {df['hr'].mean():.1f} ± {df['hr'].std():.1f}")
    log.info(f"  Average SpO2: {df['spo2'].mean():.1f} ± {df['spo2'].std():.1f}")
    
    return df

def load_patient_specific_data(patient_id, full_df):
    """
    Loads specific data for a patient.
    Filters dataset data by corresponding patient_id.
    """
    if 'patient_id' in full_df.columns:
        # Convert patient_id to string for comparison
        patient_id_str = str(patient_id)
        patient_df = full_df[full_df['patient_id'].astype(str) == patient_id_str]
        
        if len(patient_df) > 0:
            log.info(f"  ID {patient_id}: {len(patient_df)} samples | "
                    f"HR: {patient_df['hr'].mean():.1f}±{patient_df['hr'].std():.1f} | "
                    f"SpO2: {patient_df['spo2'].mean():.1f}±{patient_df['spo2'].std():.1f}")
            return patient_df.reset_index(drop=True)
        else:
            log.warning(f"  ID {patient_id}: not found in dataset")
            # Return full dataset as fallback
            return full_df
    
    # If there's no patient_id column, return full dataset
    log.warning(f"  Dataset without 'patient_id' column, all patients will use the same data")
    return full_df

# ------------- Initial Vitals Generation for High/Moderate Risk ------------
def get_initial_vitals_high_risk():
    """
    Returns random vitals that typically result in HIGH risk (NEWS2 >= 7).
    Uses combinations of abnormal values: hypotension, fever/hypothermia, tachypnea/bradypnea, etc.
    These values ensure at least 7 points from non-HR/SpO2 parameters.
    """
    scenarios = [
        # sys_bp <=90 (3pts) + temp >=39 (2pts) + rr >=25 (3pts) = 8pts minimum
        {"sys_bp": random.randint(82, 90), "temp": random.uniform(39.1, 40.0), "rr": random.randint(25, 29), "consciousness": "A"},
        # sys_bp <=90 (3pts) + rr <=8 (3pts) + consciousness=V (3pts) = 9pts minimum
        {"sys_bp": random.randint(80, 88), "temp": random.uniform(36.2, 37.8), "rr": random.randint(5, 8), "consciousness": "V"},
        # sys_bp <=100 (2pts) + temp >=39 (2pts) + rr >=25 (3pts) + consciousness=V (3pts) = 10pts minimum
        {"sys_bp": random.randint(92, 100), "temp": random.uniform(39.0, 39.8), "rr": random.randint(25, 28), "consciousness": "V"},
        # sys_bp <=90 (3pts) + temp <=35 (3pts) + rr 21-24 (2pts) = 8pts minimum
        {"sys_bp": random.randint(82, 90), "temp": random.uniform(34.5, 35.0), "rr": random.randint(21, 24), "consciousness": "A"},
    ]
    return random.choice(scenarios)

def get_initial_vitals_moderate_risk():
    """
    Returns random vitals that typically result in MODERATE risk (NEWS2 = 5-6).
    Uses milder combinations that guarantee at least 5 points from non-HR/SpO2 parameters.
    """
    scenarios = [
        # sys_bp 91-100 (2pts) + temp >=39 (2pts) + rr 21-24 (2pts) = 6pts
        {"sys_bp": random.randint(91, 100), "temp": random.uniform(39.0, 39.5), "rr": random.randint(21, 24), "consciousness": "A"},
        # sys_bp <=90 (3pts) + temp 36.1-38 (0pts) + rr 21-24 (2pts) = 5pts
        {"sys_bp": random.randint(85, 90), "temp": random.uniform(36.5, 37.5), "rr": random.randint(21, 24), "consciousness": "A"},
        # sys_bp 101-110 (1pt) + temp 35.1-36 (1pt) + rr 21-24 (2pts) + consciousness=V (3pts) = 7pts
        # Actually this is HIGH, adjust to get 5-6
        {"sys_bp": random.randint(101, 110), "temp": random.uniform(38.1, 38.9), "rr": random.randint(21, 24), "consciousness": "A"},
        # sys_bp 91-100 (2pts) + temp 35.1-36 (1pt) + rr 21-24 (2pts) = 5pts
        {"sys_bp": random.randint(91, 100), "temp": random.uniform(35.1, 36.0), "rr": random.randint(21, 24), "consciousness": "A"},
    ]
    return random.choice(scenarios)

# ------------- Algorithm 1 Parameters ------------
PARAMS={
 "HIGH":     dict(ic_fc=30,  eps_fc=2,  dc_fc=2,  ic_spo2=30,  eps_spo2=1, dc_spo2=1, t_sdt=15,  ic_max=30*60),
 "MODERATE": dict(ic_fc=120, eps_fc=5,  dc_fc=5,  ic_spo2=180, eps_spo2=1, dc_spo2=1, t_sdt=60,  ic_max=30*60),
 "LOW":      dict(ic_fc=300, eps_fc=5,  dc_fc=5,  ic_spo2=600, eps_spo2=2, dc_spo2=2, t_sdt=180, ic_max=2*3600),
 "MINIMAL":  dict(ic_fc=600, eps_fc=10, dc_fc=10, ic_spo2=900, eps_spo2=3, dc_spo2=3, t_sdt=300, ic_max=6*3600)
}

# --------------- Classes --------------------
class Patient:
    """Represents a patient being monitored in the simulation.
    
    Each Patient instance tracks vital signs from its assigned dataset,
    manages collection intervals based on current risk level, and implements
    the adaptive backoff algorithm for stable signals.
    
    Attributes:
        id: Unique patient identifier.
        df: DataFrame containing patient's vital signs data.
        idx: Current position in the dataset.
        news2: Latest NEWS2 score from fog layer.
        risk: Current risk classification (HIGH/MODERATE/LOW/MINIMAL).
        spo2_scale: SpO2 scoring scale (1=standard, 2=COPD).
        on_o2: Whether patient is receiving supplemental oxygen.
        base_consciousness: AVPU consciousness level.
        ic_fc, ic_spo2: Collection intervals for HR and SpO2 (seconds).
        eps_fc, eps_spo2: Tolerance for backoff detection.
        dc_fc, dc_spo2: SDT deviation parameters.
        t_sdt: SDT maximum time gap parameter.
        persistent_high_risk: If True, patient never drops below HIGH risk.
    
    The patient's sampling parameters are automatically updated when
    the fog layer returns new NEWS2 scores via update_risk().
    """
    
    def __init__(self, pid, df, news2=0, persistent_high_risk=False, force_high_risk=False,
                 spo2_scale=None, on_o2=None):
        self.id=pid; self.df=df; self.idx=random.randrange(len(df))
        self.persistent_high_risk=persistent_high_risk
        self.force_high_risk=force_high_risk
        self.scenario_step=0
        
        # Configure patient-specific parameters from config
        self._configure_patient_params(spo2_scale, on_o2)
        
        self.update_risk(news2)
        self.last_sent_fc=self._cur('hr'); self.last_sent_spo2=self._cur('spo2')
        self.fc_buf,self.spo2_buf=[],[]
        self.last_fc_col=self.last_spo2_col=time.time()
        self.stable_fc=self.stable_spo2=0
    
    def _configure_patient_params(self, spo2_scale, on_o2):
        """Configure patient-specific parameters from config file or random assignment."""
        # Check for patient-specific overrides first
        overrides = CONFIG.get("patient_overrides", {}).get(str(self.id), {})
        
        # Get dataset-specific probabilities
        dataset_config = CONFIG.get("datasets", {}).get(DATASET_TYPE, {})
        probs = dataset_config.get("probabilities", {})
        defaults = CONFIG.get("defaults", {})
        
        # SpO2 Scale: override > explicit param > random based on probability > default
        if "spo2_scale" in overrides:
            self.spo2_scale = overrides["spo2_scale"]
        elif spo2_scale is not None:
            self.spo2_scale = spo2_scale
        elif random.random() < probs.get("spo2_scale_2", 0):
            self.spo2_scale = 2
        else:
            self.spo2_scale = defaults.get("spo2_scale", 1)
        
        # On O2: override > explicit param > random based on probability > default
        if "on_o2" in overrides:
            self.on_o2 = overrides["on_o2"]
        elif on_o2 is not None:
            self.on_o2 = on_o2
        elif random.random() < probs.get("on_o2", 0):
            self.on_o2 = True
        else:
            self.on_o2 = defaults.get("on_o2", False)
        
        # Consciousness: only altered for high_risk with probability, unless overridden
        if "consciousness" in overrides:
            self.base_consciousness = overrides["consciousness"]
        elif random.random() < probs.get("altered_consciousness", 0):
            options = dataset_config.get("consciousness_options", ["V"])
            self.base_consciousness = random.choice(options)
        else:
            self.base_consciousness = defaults.get("consciousness", "A")
    
    def _cur(self,c):
        val = self.df.iloc[self.idx][c]
        return float(val) if pd.notna(val) else 0.0
    def next(self):
        row=self.df.iloc[self.idx]; self.idx=(self.idx+1)%len(self.df)
        hr_val = row['hr'] if pd.notna(row['hr']) else 75.0
        spo2_val = row['spo2'] if pd.notna(row['spo2']) else 98.0
        return {"timestamp":time.time(),"hr":float(hr_val),"spo2":float(spo2_val)}

    def update_risk(self,score):
        # Persistent HIGH risk patients never drop below NEWS2=7
        if self.persistent_high_risk and score<7: 
            score=7
        # For high_risk dataset: non-persistent patients should stay MODERATE or HIGH (never LOW/MINIMAL)
        elif DATASET_TYPE == "high_risk" and not self.persistent_high_risk and score<5:
            score=5  # Minimum MODERATE risk for variable patients
        old=getattr(self,'risk','N/A')
        self.news2=score
        self.risk=('HIGH' if score>=7 else 'MODERATE' if score>=5 else 'LOW' if score>=1 else 'MINIMAL')
        if self.risk!=old:
            log.info(f"[RISK] {self.id}: {old} → {self.risk} (NEWS2={score})")
        p=PARAMS[self.risk]
        self.ic_fc,self.ic_spo2=p['ic_fc'],p['ic_spo2']
        self.eps_fc,self.eps_spo2=p['eps_fc'],p['eps_spo2']
        self.dc_fc,self.dc_spo2=p['dc_fc'],p['dc_spo2']
        self.t_sdt,self.ic_max=p['t_sdt'],p['ic_max']

    def backoff(self,sig,latest):
        if self.risk=='HIGH':return
        eps=self.eps_fc if sig=='hr' else self.eps_spo2
        st_attr='stable_fc' if sig=='hr' else 'stable_spo2'
        ic_attr='ic_fc' if sig=='hr' else 'ic_spo2'
        last_attr='last_sent_fc' if sig=='hr' else 'last_sent_spo2'
        st=getattr(self,st_attr); last=getattr(self,last_attr)
        if abs(latest-last)<=eps: st+=1
        else: st=0; setattr(self,ic_attr,PARAMS[self.risk][ic_attr])
        if st>=K_STABLE:
            cur=getattr(self,ic_attr); new=min(cur*2,self.ic_max)
            if new>cur: log.info(f"[BACKOFF] {self.id} {sig.upper()} {cur}s→{new}s")
            setattr(self,ic_attr,new); st=0
        setattr(self,st_attr,st); setattr(self,last_attr,latest)

    def get_current_vitals(self):
        """
        Returns a complete vitals dictionary for NEWS2 calculation.
        Includes HR and SpO2 from dataset, plus other parameters based on risk profile.
        Uses patient-specific configuration for spo2_scale and on_o2.
        """
        # Get current HR and SpO2 from dataset
        current_hr = self._cur('hr')
        current_spo2 = self._cur('spo2')
        
        # Base vitals with dataset values and patient-specific params
        vitals = {
            "hr": current_hr,
            "spo2": current_spo2,
            "spo2_scale": self.spo2_scale,
            "on_o2": self.on_o2
        }
        
        # For high_risk dataset: add forced parameters to ensure MODERATE or HIGH
        if DATASET_TYPE == "high_risk":
            if self.force_high_risk:
                # Persistent HIGH patients get vitals that ensure NEWS2 >= 7
                forced = get_initial_vitals_high_risk()
            else:
                # Variable patients get vitals that ensure NEWS2 = 5-6 (MODERATE)
                forced = get_initial_vitals_moderate_risk()
            # Use patient's base consciousness unless forced vitals override it
            if "consciousness" not in forced or self.base_consciousness != "A":
                forced["consciousness"] = self.base_consciousness
            vitals.update(forced)
        else:
            # For low_risk dataset: use normal vitals with patient-specific consciousness
            vitals.update({
                "sys_bp": 120,
                "temp": 36.5,
                "rr": 18,
                "consciousness": self.base_consciousness
            })
        
        return vitals

class Assembler:
    """Batches compressed data packets by risk level for transmission.
    
    The Assembler implements risk-based packet aggregation with configurable
    thresholds for size and timeout. Higher-risk data is transmitted more
    frequently with smaller batches to ensure timely delivery.
    
    Configuration per risk level:
        - HIGH: 15s timeout, 5KB size limit
        - MODERATE: 60s timeout, 20KB size limit  
        - LOW: 150s timeout, 50KB size limit
        - MINIMAL: 300s timeout, 50KB size limit
    
    Methods:
        add(pkg, risk): Add a compressed packet to the appropriate queue.
        get_ready_batches(): Return list of batches that should be sent
            (either due to size limit or timeout reached).
    
    Attributes:
        queues: Dict mapping risk levels to their queue state
            (buffer, size, timestamp, config).
    """
    
    def __init__(self, cfg):
        """Initialize Assembler with configuration for each risk level.
        
        Args:
            cfg: Dict mapping risk levels to {timeout, size_limit} dicts.
        """
        self.queues = {r: {"buffer": [], "size": 0, "timestamp": None, "config": conf} 
                       for r, conf in cfg.items()}
    
    def add(self, pkg, risk):
        """Add a data packet to the queue for the specified risk level.
        
        Args:
            pkg: Dict containing compressed data and metadata.
            risk: Risk level string (HIGH/MODERATE/LOW/MINIMAL).
        """
        q = self.queues[risk]
        if not q['buffer']:
            q['timestamp'] = time.time()
        q['buffer'].append(pkg)
        q['size'] += pkg['post_sdt_size']
        log.info(f"  [Pack] +{pkg['post_sdt_size']}b → queue {risk}={q['size']}b")
    
    def get_ready_batches(self):
        """Check all queues and return batches that are ready to send.
        
        A batch is ready when either:
            1. Size limit is exceeded, or
            2. Timeout has been reached
        
        Returns:
            List of dicts with keys: batch (list of packets), risk (str),
            reason ('size' or 'timeout').
        """
        ready = []
        for r, q in self.queues.items():
            if not q['buffer']:
                continue
            if q['size'] >= q['config']['size_limit'] or \
               (q['timestamp'] and time.time() - q['timestamp'] >= q['config']['timeout']):
                reason = "size" if q['size'] >= q['config']['size_limit'] else "timeout"
                ready.append({"batch": q['buffer'], "risk": r, "reason": reason})
                q['buffer'], q['size'], q['timestamp'] = [], 0, None
        return ready

def lossless(txt: str, risk: str):
    """Apply lossless compression to payload based on size and risk.
    
    Compression strategy:
        - HIGH risk: No compression (latency priority)
        - Payload < 1KB: No compression (overhead not worth it)
        - Payload < 32KB: Huffman coding
        - Payload >= 32KB: LZW compression
    
    Args:
        txt: JSON string payload to compress.
        risk: Patient risk level.
    
    Returns:
        Tuple of (compressed_bytes, compression_type_header).
        Header is 'none', 'hushman' (Huffman), or 'lzw'.
    
    Note:
        'hushman' is a legacy header name for Huffman compression,
        maintained for backward compatibility with older fog versions.
    """
    raw = txt.encode()
    size = len(raw)
    if risk == 'HIGH' or size < HUFF_MIN:
        return raw, 'none'
    if size < LZW_MIN:
        return json.dumps(Huffman().compress(txt)).encode(), 'hushman'
    return json.dumps(LZW().compress(txt)).encode(), 'lzw'

def send_batch(batch_info):
    """Send a batch of compressed data to the fog layer.
    
    Supports both HTTP POST and MQTT protocols based on EDGE_USE_MQTT
    environment variable. Applies lossless compression if appropriate
    and tracks latency metrics.
    
    Args:
        batch_info: Dict with keys:
            - batch: List of compressed data packets
            - risk: Risk level string
            - reason: Why batch was triggered ('size' or 'timeout')
    
    Returns:
        Dict mapping patient IDs to their new NEWS2 scores from fog response,
        or empty dict if transmission failed.
    
    Side Effects:
        - Updates LATENCY_METRICS global with transmission timing
        - Logs transmission details and response
    """
    batch=batch_info['batch']; risk=batch_info['risk']
    total_raw_size = sum(p.get('raw_size', 0) for p in batch)
    payload_str=json.dumps(batch)
    raw_payload_size = len(payload_str.encode())
    payload, hdr = lossless(payload_str, risk)

    if hdr == 'none':
        if risk == 'HIGH':
            log.info(f"  [STEP 2] Lossless Compression skipped (HIGH Risk).")
        elif raw_payload_size < HUFF_MIN:
            log.info(f"  [STEP 2] Lossless Compression skipped (Batch {raw_payload_size}b < threshold {HUFF_MIN}b).")
    else:
        log.info(f"  [STEP 2] Lossless Compression applied ({hdr}).")

    final_size=len(payload)
    start=time.time()
    # MQTT option: set EDGE_USE_MQTT=1 and MQTT_BROKER / MQTT_PORT if needed
    use_mqtt = os.environ.get('EDGE_USE_MQTT','0') in ('1','true','True')
    if use_mqtt:
        try:
            import paho.mqtt.client as mqtt
            resp_topic = f"vispac/resp/{uuid.uuid4()}"
            msg = json.dumps({
                'edge_id': EDGE_ID,
                'reply_topic': resp_topic, 
                'X-Compression-Type': hdr, 
                'payload': payload.decode(),
                'send_timestamp': time.time()
            })

            q = []
            received = {'data': None}
            def _on_message(client, userdata, message):
                try:
                    received['data'] = json.loads(message.payload.decode())
                except Exception:
                    received['data'] = None

            client = mqtt.Client()
            client.on_message = _on_message
            broker = os.environ.get('MQTT_BROKER','127.0.0.1')
            port = int(os.environ.get('MQTT_PORT','1883'))
            client.connect(broker, port, 60)
            client.loop_start()
            client.subscribe(resp_topic)
            client.publish('vispac/upload_batch', msg)

            # wait for response (max 10s)
            timeout = 10
            waited = 0
            while waited < timeout and received['data'] is None:
                time.sleep(0.1); waited += 0.1
            client.loop_stop(); client.disconnect()
            resp = received['data'] or {}
            elapsed=time.time()-start
            latency_ms = elapsed * 1000
            
            # Update latency metrics
            LATENCY_METRICS["send_count"] += 1
            LATENCY_METRICS["total_latency_ms"] += latency_ms
            LATENCY_METRICS["min_latency_ms"] = min(LATENCY_METRICS["min_latency_ms"], latency_ms)
            LATENCY_METRICS["max_latency_ms"] = max(LATENCY_METRICS["max_latency_ms"], latency_ms)
            LATENCY_METRICS["by_risk"][risk].append(latency_ms)
            
            ratio=100*final_size/total_raw_size if total_raw_size>0 else 0
            log.info(f"[MQTT SEND] {EDGE_ID} | Batch '{risk}' | {len(batch)} pkts | {total_raw_size}b→{final_size}b ({ratio:.1f}%) | {hdr} | {latency_ms:.1f}ms | scores={resp.get('scores')}")
            return resp.get('scores',{})
        except Exception as e:
            log.error(f"Failed to send batch via MQTT: {e}")
            return {}

    try:
        headers = {
            'X-Compression-Type': hdr,
            'Content-Type': 'application/json',
            'X-Edge-ID': EDGE_ID,
            'X-Send-Timestamp': str(time.time())
        }
        r=requests.post(API_URL, data=payload, headers=headers, timeout=10)
        r.raise_for_status(); resp=r.json()
        elapsed=time.time()-start
        latency_ms = elapsed * 1000
        
        # Update latency metrics
        LATENCY_METRICS["send_count"] += 1
        LATENCY_METRICS["total_latency_ms"] += latency_ms
        LATENCY_METRICS["min_latency_ms"] = min(LATENCY_METRICS["min_latency_ms"], latency_ms)
        LATENCY_METRICS["max_latency_ms"] = max(LATENCY_METRICS["max_latency_ms"], latency_ms)
        LATENCY_METRICS["by_risk"][risk].append(latency_ms)
        
        ratio=100*final_size/total_raw_size if total_raw_size>0 else 0
        log.info(f"[SEND] {EDGE_ID} | Batch '{risk}' | {len(batch)} pkts | {total_raw_size}b→{final_size}b ({ratio:.1f}%) | {hdr} | {latency_ms:.1f}ms | scores={resp.get('scores')}")
        return resp.get('scores',{})
    except Exception as e:
        log.error(f"Failed to send batch: {e}")
        return {}

# --------------- MAIN LOOP ----------------
def main():
    full_df=ensure_dataset()
    
    # Detect available IDs in dataset
    if 'patient_id' in full_df.columns:
        available_ids = sorted(full_df['patient_id'].unique())
        
        # Use PATIENT_RANGE to select which patients this edge handles
        selected_ids = parse_patient_range(PATIENT_RANGE, available_ids)
        
        # Legacy NUM_PATIENTS support (if PATIENT_RANGE not set and NUM_PATIENTS > 0)
        if PATIENT_RANGE.lower() == "all" and NUM_PATIENTS > 0:
            selected_ids = selected_ids[:NUM_PATIENTS]
        
        patient_ids = [str(pid) for pid in selected_ids]
        log.info(f"IDs detected in dataset: {len(available_ids)} unique")
        log.info(f"PATIENT_RANGE: '{PATIENT_RANGE}' → selected {len(patient_ids)} patients: {patient_ids}")
    else:
        # Fallback to generic IDs if there's no patient_id column
        patient_ids = ["PID-001", "PID-002", "PID-003", "PID-004", "PID-005", "PID-006", "PID-007"]
        log.warning("Dataset without 'patient_id' column, using generic IDs")
    
    patients = []
    
    log.info("="*60)
    log.info(f"VISPAC SIMULATION STARTED")
    log.info(f"  Edge ID: {EDGE_ID}")
    log.info(f"  Dataset: {DATASET_TYPE.upper()}")
    log.info(f"  Patients: {len(patient_ids)} ({patient_ids[0]} to {patient_ids[-1]})")
    log.info("="*60)
    log.info("Loading patient data:")
    
    for i, pid in enumerate(patient_ids):
        patient_df = load_patient_specific_data(pid, full_df)
        # Force HIGH risk for high_risk dataset to simulate critical patients
        # Only force half of the patients to persistent HIGH risk (alternating)
        force_high = (DATASET_TYPE == "high_risk") and (i % 2 == 0)  # Even index patients stay HIGH
        
        # Start with a nominal NEWS2 score that will be updated by fog on first batch
        # For high_risk: use higher initial scores to ensure proper risk category assignment
        # For low_risk: start at 0 (MINIMAL) since data represents healthy individuals
        if DATASET_TYPE == "high_risk":
            initial_news2 = 7 if force_high else 5
        else:
            initial_news2 = 0  # Low risk dataset starts at MINIMAL
        
        # Create patient - config params (spo2_scale, on_o2) loaded from config file
        patient = Patient(pid, patient_df, news2=initial_news2, 
                         persistent_high_risk=force_high, force_high_risk=force_high)
        
        # Log patient configuration
        risk_type = "PERSISTENT HIGH" if force_high else "VARIABLE"
        log.info(f"  {pid}: spo2_scale={patient.spo2_scale}, on_o2={patient.on_o2}, "
                f"consciousness={patient.base_consciousness} ({risk_type})")
        
        patients.append(patient)
    
    log.info("="*60)
    
    assembler=Assembler(ASSEMBLER_CONFIG)
    sdt=SwingingDoorCompressor()

    prd_accumulator = {'hr': [], 'spo2': []}

    last_keep=time.time()
    
    try:
        while True:
            now=time.time()
            if now-last_keep>=KEEP_ALIVE:
                log.info("[ALG3] Keep‑alive check …")
                for p in patients:
                    v=p.next()
                    if abs(v['hr']-p.last_sent_fc)>p.eps_fc: p.last_fc_col=0
                    if abs(v['spo2']-p.last_sent_spo2)>p.eps_spo2: p.last_spo2_col=0
                last_keep=now
            for p in patients:
                v=p.next(); p.fc_buf.append((v['timestamp'],v['hr'])); p.spo2_buf.append((v['timestamp'],v['spo2']))
                
                if now-p.last_fc_col>=p.ic_fc:
                    raw_size=len(json.dumps(p.fc_buf).encode())
                    comp=sdt.compress(p.fc_buf,p.dc_fc,p.t_sdt)
                    if comp:
                        post_sdt_size=len(json.dumps(comp).encode())
                        reduction=100*(1-post_sdt_size/raw_size) if raw_size>0 else 0
                        log.info(f"[HR COLLECT] {p.id} {p.risk} | {len(p.fc_buf)}→{len(comp)} pts | {raw_size}b→{post_sdt_size}b ({reduction:.1f}%)")
                        
                        original_signal_values = [point[1] for point in p.fc_buf]
                        reconstructed_signal = reconstruct_signal(p.fc_buf, comp)
                        prd_value = calculate_prd(original_signal_values, reconstructed_signal)
                        prd_accumulator['hr'].append(prd_value)
                        log.info(f"  [HR DISTORTION] {p.id} PRD={prd_value:.4f}%")

                        vitals=p.get_current_vitals()
                        entry={'patient_id':p.id,'signal':'hr','risco':p.risk,'data':comp, 'raw_size':raw_size, 'post_sdt_size':post_sdt_size, 'vitals':vitals}
                        assembler.add(entry,p.risk)
                    p.fc_buf=[]; p.last_fc_col=now; p.backoff('hr',v['hr'])
                
                if now-p.last_spo2_col>=p.ic_spo2:
                    raw_size=len(json.dumps(p.spo2_buf).encode())
                    comp=sdt.compress(p.spo2_buf,p.dc_spo2,p.t_sdt)
                    if comp:
                        post_sdt_size=len(json.dumps(comp).encode())
                        reduction=100*(1-post_sdt_size/raw_size) if raw_size>0 else 0
                        log.info(f"[SpO2 COLLECT] {p.id} {p.risk} | {len(p.spo2_buf)}→{len(comp)} pts | {raw_size}b→{post_sdt_size}b ({reduction:.1f}%)")
                        
                        original_signal_values = [point[1] for point in p.spo2_buf]
                        reconstructed_signal = reconstruct_signal(p.spo2_buf, comp)
                        prd_value = calculate_prd(original_signal_values, reconstructed_signal)
                        prd_accumulator['spo2'].append(prd_value)
                        log.info(f"  [SpO2 DISTORTION] {p.id} PRD={prd_value:.4f}%")

                        vitals=p.get_current_vitals()
                        entry={'patient_id':p.id,'signal':'spo2','risco':p.risk,'data':comp, 'raw_size':raw_size, 'post_sdt_size':post_sdt_size, 'vitals':vitals}
                        assembler.add(entry,p.risk)
                    p.spo2_buf=[]; p.last_spo2_col=now; p.backoff('spo2',v['spo2'])
            
            for b in assembler.get_ready_batches():
                log.info(f"[Pack] batch queue '{b['risk']}' ready by {b['reason']}")
                feedback=send_batch(b)
                if feedback:
                    for pid,score in feedback.items():
                        for p in patients:
                            if p.id==pid: p.update_risk(score)
            
            queue_sizes = " | ".join([f"{r[:3]}:{q['size']}b" for r,q in assembler.queues.items()])
            print(f"\r[{time.strftime('%H:%M:%S')}] Monitoring... Queues: [ {queue_sizes} ]", end="")
            time.sleep(0.5)

    finally:
        log.info("="*50)
        log.info(f"Simulation ended for Edge {EDGE_ID}")
        log.info("PRD statistics for all patients:")
        if prd_accumulator['hr']:
            avg_hr = np.mean(prd_accumulator['hr'])
            min_hr = np.min(prd_accumulator['hr'])
            max_hr = np.max(prd_accumulator['hr'])
            log.info(f"  HR (hr): Mean={avg_hr:.4f}% | Min={min_hr:.4f}% | Max={max_hr:.4f}%")
        if prd_accumulator['spo2']:
            avg_spo2 = np.mean(prd_accumulator['spo2'])
            min_spo2 = np.min(prd_accumulator['spo2'])
            max_spo2 = np.max(prd_accumulator['spo2'])
            log.info(f"  SpO2:    Mean={avg_spo2:.4f}% | Min={min_spo2:.4f}% | Max={max_spo2:.4f}%")
        log.info("-"*50)
        log_latency_summary()
        log.info("="*50)


if __name__=='__main__':
    main()