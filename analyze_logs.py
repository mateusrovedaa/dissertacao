#!/usr/bin/env python3
"""
VISPAC Log Analysis Script with HTML Dashboard

Parses experiment logs from edge devices and extracts metrics for comparison
across the 3 test scenarios (baseline, static, vispac).

Features:
- Metrics extraction (compression, PRD, latency)
- Risk evolution tracking for scenario 3
- Interactive HTML dashboard with charts

Usage:
    python analyze_logs.py logs/scenario1_baseline --output results/
    python analyze_logs.py --compare logs/ --output results/

Author: VISPAC Research
"""

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class LogEntry:
    """Single log entry with extracted metrics."""
    timestamp: datetime
    edge_id: str
    patient_id: str
    signal: str  # 'hr' or 'spo2'
    risk: str
    raw_size: int = 0
    compressed_size: int = 0
    compression_ratio: float = 0.0
    prd: float = 0.0
    latency_ms: float = 0.0


@dataclass
class RiskChangeEvent:
    """Records a patient risk level change."""
    timestamp: datetime
    patient_id: str
    old_risk: str
    new_risk: str
    news2_score: int


@dataclass
class EdgeMetrics:
    """Aggregated metrics for a single edge device."""
    edge_id: str
    scenario: str
    total_transmissions: int = 0
    total_raw_bytes: int = 0
    total_compressed_bytes: int = 0
    compression_ratios: List[float] = field(default_factory=list)
    prd_values: List[float] = field(default_factory=list)
    latencies: List[float] = field(default_factory=list)
    risk_distribution: Dict[str, int] = field(default_factory=dict)
    risk_events: List[RiskChangeEvent] = field(default_factory=list)
    # Patient-level risk history: patient_id -> [(timestamp, risk, news2_score)]
    patient_risk_history: Dict[str, List[Tuple[datetime, str, int]]] = field(default_factory=dict)
    # All NEWS2 scores per patient: patient_id -> [(timestamp, news2_score)]
    patient_news2_history: Dict[str, List[Tuple[datetime, int]]] = field(default_factory=dict)
    # Backoff events per patient: patient_id -> [(timestamp, signal, old_interval, new_interval)]
    patient_backoff_history: Dict[str, List[Tuple[datetime, str, int, int]]] = field(default_factory=dict)
    # Resource metrics
    cpu_samples: List[float] = field(default_factory=list)
    memory_samples: List[float] = field(default_factory=list)
    
    @property
    def avg_compression(self) -> float:
        return sum(self.compression_ratios) / len(self.compression_ratios) if self.compression_ratios else 0.0
    
    @property
    def avg_prd(self) -> float:
        return sum(self.prd_values) / len(self.prd_values) if self.prd_values else 0.0
    
    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
    
    @property
    def std_latency(self) -> float:
        if len(self.latencies) < 2:
            return 0.0
        avg = self.avg_latency
        variance = sum((x - avg) ** 2 for x in self.latencies) / len(self.latencies)
        return variance ** 0.5
    
    @property
    def avg_memory(self) -> float:
        return sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0.0


@dataclass
class FogMetrics:
    """Aggregated metrics for Fog Layer."""
    scenario: str
    total_batches: int = 0
    total_forwarded: int = 0
    # Throughput (timestamps of processed batches)
    batch_timestamps: List[datetime] = field(default_factory=list)
    # Performance
    process_latencies: List[float] = field(default_factory=list)
    forward_latencies: List[float] = field(default_factory=list)
    
    @property
    def avg_process_latency(self) -> float:
        return sum(self.process_latencies) / len(self.process_latencies) if self.process_latencies else 0.0

    @property
    def avg_forward_latency(self) -> float:
        return sum(self.forward_latencies) / len(self.forward_latencies) if self.forward_latencies else 0.0


@dataclass
class CloudMetrics:
    """Aggregated metrics for Cloud Layer."""
    scenario: str
    total_items_stored: int = 0
    # Throughput
    insert_timestamps: List[datetime] = field(default_factory=list)
    # Performance
    insert_latencies: List[float] = field(default_factory=list)
    
    @property
    def avg_insert_latency(self) -> float:
        return sum(self.insert_latencies) / len(self.insert_latencies) if self.insert_latencies else 0.0


class LogParser:
    """Parses VISPAC edge log files and extracts metrics."""
    
    # Regex patterns for log parsing
    PATTERNS = {
        # [HR COLLECT] 6 HIGH | 120→45 pts | 2400b→900b (62.5%)
        'collect': re.compile(
            r'\[(HR|SpO2) COLLECT\]\s+(\S+)\s+(\w+)\s+\|.*?\|.*?(\d+)b→(\d+)b\s+\((\d+\.?\d*)%\)'
        ),
        # [HR COLLECT] 6 HIGH | 120 pts | 2400b (RAW - no compression)
        'collect_raw': re.compile(
            r'\[(HR|SpO2) COLLECT\]\s+(\S+)\s+(\w+)\s+\|.*?\|.*?(\d+)b\s+\(RAW'
        ),
        # [HR DISTORTION] 6 PRD=0.0234%
        'prd': re.compile(
            r'\[(HR|SpO2) DISTORTION\]\s+(\S+)\s+PRD=(\d+\.?\d*)%'
        ),
        # [MQTT SEND] edge-01 | Batch 'HIGH' | 5 pkts | 1234b→567b (54.1%) | lzw | 12.5ms
        'send': re.compile(
            r'\[(MQTT )?SEND\]\s+(\S+)\s+\|.*?(\d+)b→(\d+)b.*?\|.*?(\d+\.?\d*)ms'
        ),
        # [RISK] 6: LOW → HIGH (NEWS2=8) [DYNAMIC] or [RISK] 26: N/A → HIGH (NEWS2=7) [DYNAMIC]
        'risk': re.compile(
            r'\[RISK\]\s+(\S+):\s+(\S+)\s+→\s+(\w+)\s+\(NEWS2=(\d+)\)'
        ),
        # Timestamp at start of line
        'timestamp': re.compile(
            r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d+)'
        ),
        # [RESOURCE SUMMARY] - CPU and Memory
        'resource_cpu': re.compile(
            r'CPU:\s+Avg=(\d+\.?\d*)%'
        ),
        'resource_mem': re.compile(
            r'Memory:\s+Avg=(\d+\.?\d*)MB'
        ),
        # scores={'14': 11, '26': 8, '24': 9} at end of MQTT SEND
        'scores': re.compile(
            r"scores=\{([^}]+)\}"
        ),
        # [BACKOFF] 6 HR 300s→600s
        'backoff': re.compile(
            r'\[BACKOFF\]\s+(\S+)\s+(HR|SPO2)\s+(\d+)s→(\d+)s'
        ),
        # Fog/Cloud Metrics (New)
        'fog_metrics': re.compile(
            r'\[FOG_METRICS\]\s+type=(\S+)\s+batch_size=(\d+)\s+process_ms=(\d+\.?\d*)\s+forward_ms=(\d+\.?\d*)'
        ),
        'cloud_metrics': re.compile(
            r'\[CLOUD_METRICS\]\s+risk=(\S+)\s+items=(\d+)\s+insert_ms=(\d+\.?\d*)'
        ),
        # Fog/Cloud Metrics (Legacy/Fallback)
        'fog_process': re.compile(r'Processed batch: (\d+) patients'),
        'fog_forward': re.compile(r'Forwarding (\d+) items to cloud'),
        'cloud_store': re.compile(r'Successfully stored (\d+) items to DB'),
    }
    
    def __init__(self, scenario: str):
        self.scenario = scenario
        self.metrics: Dict[str, EdgeMetrics] = {}
        self.fog_metrics = FogMetrics(scenario=scenario)
        self.cloud_metrics = CloudMetrics(scenario=scenario)
        self.current_timestamp: Optional[datetime] = None
    
    def parse_file(self, filepath: Path) -> None:
        """Parse a single log file."""
        edge_id = None
        log_type = "other"
        
        # Determine edge_id and log type from filename
        filename = filepath.name
        if "fog" in filename:
            log_type = "fog"
        elif "cloud" in filename:
            log_type = "cloud"
        elif "edge-" in filename:
            log_type = "edge"
            edge_match = re.search(r'edge-(\d+)', filename)
            if edge_match:
                edge_id = f"edge-{edge_match.group(1)}"
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                self._parse_line(line, edge_id, log_type)
    
    def _parse_line(self, line: str, edge_id: Optional[str], log_type: str = "other") -> None:
        """Parse a single line from the log file."""
        # Extract timestamp (common to all logs)
        ts_match = self.PATTERNS['timestamp'].search(line)
        if ts_match:
            try:
                self.current_timestamp = datetime.strptime(
                    ts_match.group(1), '%Y-%m-%d %H:%M:%S,%f'
                )
            except ValueError:
                pass
        
        if log_type == "fog":
            # New Fog Metrics
            match = self.PATTERNS['fog_metrics'].search(line)
            if match:
                type_, batch_size, proc_ms, fwd_ms = match.groups()
                self.fog_metrics.total_batches += 1
                if self.current_timestamp:
                    self.fog_metrics.batch_timestamps.append(self.current_timestamp)
                self.fog_metrics.process_latencies.append(float(proc_ms))
                self.fog_metrics.forward_latencies.append(float(fwd_ms))
                return
            
            # Legacy Fog
            match = self.PATTERNS['fog_process'].search(line)
            if match:
                self.fog_metrics.total_batches += 1
                if self.current_timestamp:
                    self.fog_metrics.batch_timestamps.append(self.current_timestamp)
                return
            return 

        if log_type == "cloud":
            # New Cloud Metrics
            match = self.PATTERNS['cloud_metrics'].search(line)
            if match:
                risk, items, ins_ms = match.groups()
                self.cloud_metrics.total_items_stored += int(items)
                if self.current_timestamp:
                    self.cloud_metrics.insert_timestamps.append(self.current_timestamp)
                self.cloud_metrics.insert_latencies.append(float(ins_ms))
                return
            
            # Legacy Cloud
            match = self.PATTERNS['cloud_store'].search(line)
            if match:
                items = int(match.group(1))
                self.cloud_metrics.total_items_stored += items
                if self.current_timestamp:
                    self.cloud_metrics.insert_timestamps.append(self.current_timestamp)
                return
            return

        if log_type == "edge":
            if not edge_id:
                return  # Skip edge-specific parsing if not an edge log

        # Ensure metrics object exists for this edge
        if edge_id not in self.metrics:
            self.metrics[edge_id] = EdgeMetrics(edge_id=edge_id, scenario=self.scenario)
        
        metrics = self.metrics[edge_id]
        
        # Try to match compression with ratio
        match = self.PATTERNS['collect'].search(line)
        if match:
            signal, patient_id, risk, raw_size, compressed_size, ratio = match.groups()
            metrics.total_transmissions += 1
            metrics.total_raw_bytes += int(raw_size)
            metrics.total_compressed_bytes += int(compressed_size)
            metrics.compression_ratios.append(float(ratio))
            metrics.risk_distribution[risk] = metrics.risk_distribution.get(risk, 0) + 1
            return
        
        # Try to match raw collection (scenario 1)
        match = self.PATTERNS['collect_raw'].search(line)
        if match:
            signal, patient_id, risk, raw_size = match.groups()
            metrics.total_transmissions += 1
            metrics.total_raw_bytes += int(raw_size)
            metrics.total_compressed_bytes += int(raw_size)  # No compression
            metrics.risk_distribution[risk] = metrics.risk_distribution.get(risk, 0) + 1
            return
        
        # Try to match PRD
        match = self.PATTERNS['prd'].search(line)
        if match:
            signal, patient_id, prd = match.groups()
            metrics.prd_values.append(float(prd))
            return
        
        # Try to match send latency and scores
        match = self.PATTERNS['send'].search(line)
        if match:
            groups = match.groups()
            latency = float(groups[-1])  # Last group is always latency
            metrics.latencies.append(latency)
            
            # Also extract NEWS2 scores from this send
            scores_match = self.PATTERNS['scores'].search(line)
            if scores_match:
                scores_str = scores_match.group(1)
                # Parse scores like '14': 11, '26': 8
                import ast
                try:
                    scores_dict = ast.literal_eval('{' + scores_str + '}')
                    for patient_id, score in scores_dict.items():
                        patient_id = str(patient_id)
                        if patient_id not in metrics.patient_news2_history:
                            metrics.patient_news2_history[patient_id] = []
                        metrics.patient_news2_history[patient_id].append(
                            (self.current_timestamp or datetime.now(), score)
                        )
                except:
                    pass
            return
        
        # Try to match risk change (scenario 3)
        match = self.PATTERNS['risk'].search(line)
        if match:
            patient_id, old_risk, new_risk, news2_score = match.groups()
            event = RiskChangeEvent(
                timestamp=self.current_timestamp or datetime.now(),
                patient_id=patient_id,
                old_risk=old_risk,
                new_risk=new_risk,
                news2_score=int(news2_score)
            )
            metrics.risk_events.append(event)
            
            # Update patient risk history
            if patient_id not in metrics.patient_risk_history:
                metrics.patient_risk_history[patient_id] = []
            metrics.patient_risk_history[patient_id].append(
                (event.timestamp, new_risk, event.news2_score)
            )
            return
        
        # Try to match resource metrics
        match = self.PATTERNS['resource_cpu'].search(line)
        if match:
            metrics.cpu_samples.append(float(match.group(1)))
            return
        
        match = self.PATTERNS['resource_mem'].search(line)
        if match:
            metrics.memory_samples.append(float(match.group(1)))
            return
        
        # Try to match backoff events (only in vispac scenario)
        if 'vispac' in self.scenario:
            match = self.PATTERNS['backoff'].search(line)
            if match:
                patient_id, signal, old_interval, new_interval = match.groups()
                if patient_id not in metrics.patient_backoff_history:
                    metrics.patient_backoff_history[patient_id] = []
                metrics.patient_backoff_history[patient_id].append(
                    (self.current_timestamp or datetime.now(), signal, int(old_interval), int(new_interval))
                )
                return
    
    def parse_directory(self, directory: Path) -> None:
        """Parse all log files in a directory."""
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Include edge, fog, and cloud logs
        log_files = (
            list(directory.glob('edge-*_error.log')) + 
            list(directory.glob('edge-*_service.log')) +
            list(directory.glob('fog_*.log')) +
            list(directory.glob('cloud_*.log'))
        )
        # Remove duplicates and keep unique files
        log_files = list(set(log_files))
        
        if not log_files:
            print(f"Warning: No log files found in {directory}")
            return
        
        print(f"Parsing {len(log_files)} log files from {directory}...")
        for filepath in sorted(log_files):
            self.parse_file(filepath)
    
    def get_summary(self) -> Dict:
        """Generate summary statistics across all edges."""
        if not self.metrics:
            return {}
        
        all_compression = []
        all_prd = []
        all_latency = []
        all_cpu = []
        all_memory = []
        total_raw = 0
        total_compressed = 0
        total_transmissions = 0
        total_risk_events = 0
        
        for m in self.metrics.values():
            all_compression.extend(m.compression_ratios)
            all_prd.extend(m.prd_values)
            all_latency.extend(m.latencies)
            all_cpu.extend(m.cpu_samples)
            all_memory.extend(m.memory_samples)
            total_raw += m.total_raw_bytes
            total_compressed += m.total_compressed_bytes
            total_transmissions += m.total_transmissions
            total_risk_events += len(m.risk_events)
        
        def safe_avg(lst):
            return sum(lst) / len(lst) if lst else 0.0
        
        def safe_std(lst):
            if len(lst) < 2:
                return 0.0
            avg = safe_avg(lst)
            variance = sum((x - avg) ** 2 for x in lst) / len(lst)
            return variance ** 0.5
        
        def safe_min(lst):
            return min(lst) if lst else 0.0
        
        def safe_max(lst):
            return max(lst) if lst else 0.0
        
        return {
            'scenario': self.scenario,
            'num_edges': len(self.metrics),
            'total_transmissions': total_transmissions,
            'total_raw_bytes': total_raw,
            'total_compressed_bytes': total_compressed,
            'overall_compression': (1 - total_compressed / total_raw) * 100 if total_raw > 0 else 0,
            'avg_compression': safe_avg(all_compression),
            'std_compression': safe_std(all_compression),
            'avg_prd': safe_avg(all_prd),
            'std_prd': safe_std(all_prd),
            'avg_latency': safe_avg(all_latency),
            'std_latency': safe_std(all_latency),
            'min_latency': safe_min(all_latency),
            'max_latency': safe_max(all_latency),
            'total_risk_events': total_risk_events,
            'avg_cpu': safe_avg(all_cpu),
            'max_cpu': safe_max(all_cpu),
            'avg_memory': safe_avg(all_memory),
            'max_memory': safe_max(all_memory),
        }
    
    def get_all_risk_histories(self) -> Dict[str, Dict[str, List]]:
        """Get complete NEWS2 score history for all patients across all edges.
        
        Uses patient_news2_history (from MQTT SEND) for complete data,
        with patient_risk_history (from RISK events) for risk labels.
        """
        all_histories = {}
        for edge_id, m in self.metrics.items():
            # Use NEWS2 history from MQTT SEND (has all scores)
            for patient_id, history in m.patient_news2_history.items():
                full_id = f"{edge_id}_{patient_id}"
                
                # Sort by timestamp
                sorted_history = sorted(history, key=lambda x: x[0])
                
                # Determine risk level from score
                def score_to_risk(score):
                    if score >= 7:
                        return 'HIGH'
                    elif score >= 5:
                        return 'MODERATE'
                    elif score >= 1:
                        return 'LOW'
                    else:
                        return 'MINIMAL'
                
                all_histories[full_id] = {
                    'timestamps': [ts.isoformat() for ts, _ in sorted_history],
                    'news2_scores': [score for _, score in sorted_history],
                    'risks': [score_to_risk(score) for _, score in sorted_history]
                }
        return all_histories
    
    def get_all_backoff_histories(self) -> Dict[str, Dict[str, List]]:
        """Get backoff event history for all patients across all edges.
        
        Returns collection interval changes over time for HR and SpO2.
        """
        all_histories = {}
        for edge_id, m in self.metrics.items():
            for patient_id, history in m.patient_backoff_history.items():
                full_id = f"{edge_id}_{patient_id}"
                
                # Sort by timestamp
                sorted_history = sorted(history, key=lambda x: x[0])
                
                all_histories[full_id] = {
                    'timestamps': [ts.isoformat() for ts, _, _, _ in sorted_history],
                    'signals': [signal for _, signal, _, _ in sorted_history],
                    'old_intervals': [old for _, _, old, _ in sorted_history],
                    'new_intervals': [new for _, _, _, new in sorted_history]
                }
        return all_histories


def export_csv(metrics: Dict[str, EdgeMetrics], output_path: Path) -> None:
    """Export metrics to CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'edge_id', 'scenario', 'total_transmissions',
            'total_raw_bytes', 'total_compressed_bytes',
            'avg_compression_%', 'avg_prd_%', 
            'avg_latency_ms', 'std_latency_ms',
            'avg_cpu_%', 'avg_memory_mb',
            'risk_high', 'risk_moderate', 'risk_low', 'risk_minimal',
            'risk_change_events'
        ])
        
        for m in metrics.values():
            writer.writerow([
                m.edge_id,
                m.scenario,
                m.total_transmissions,
                m.total_raw_bytes,
                m.total_compressed_bytes,
                f"{m.avg_compression:.2f}",
                f"{m.avg_prd:.4f}",
                f"{m.avg_latency:.2f}",
                f"{m.std_latency:.2f}",
                f"{m.avg_cpu:.1f}" if m.cpu_samples else "N/A",
                f"{m.avg_memory:.1f}" if m.memory_samples else "N/A",
                m.risk_distribution.get('HIGH', 0),
                m.risk_distribution.get('MODERATE', 0),
                m.risk_distribution.get('LOW', 0),
                m.risk_distribution.get('MINIMAL', 0),
                len(m.risk_events),
            ])
    
    print(f"CSV exported to: {output_path}")


def generate_html_dashboard(summaries: List[Dict], parsers: Dict[str, LogParser], output_path: Path) -> None:
    """Generate interactive HTML dashboard with charts."""
    
    # Prepare data for charts
    scenario_names = [s['scenario'] for s in summaries]
    compression_data = [s.get('overall_compression', 0) for s in summaries]
    prd_data = [s.get('avg_prd', 0) for s in summaries]
    latency_data = [s.get('avg_latency', 0) for s in summaries]
    cpu_data = [s.get('avg_cpu', 0) for s in summaries]
    memory_data = [s.get('avg_memory', 0) for s in summaries]
    
    # Get risk histories for scenario 3
    risk_histories = {}
    backoff_histories = {}
    
    # Extract Fog/Cloud stats per scenario
    fog_stats = []
    cloud_stats = []
    
    for s_name in scenario_names:
        parser = parsers.get(s_name)
        if parser:
            if 'vispac' in s_name:
                risk_histories = parser.get_all_risk_histories()
                backoff_histories = parser.get_all_backoff_histories()
            
            # Fog
            f = parser.fog_metrics
            step_f = max(1, len(f.batch_timestamps)//500) # limit points
            fog_stats.append({
                'scenario': s_name,
                'total_batches': f.total_batches,
                'avg_process_ms': f.avg_process_latency,
                'avg_forward_ms': f.avg_forward_latency,
                'timestamps': [t.isoformat() for t in f.batch_timestamps][::step_f],
                'latencies': f.process_latencies[::step_f]
            })
            
            # Cloud
            c = parser.cloud_metrics
            step_c = max(1, len(c.insert_timestamps)//500)
            cloud_stats.append({
                'scenario': s_name,
                'total_items': c.total_items_stored,
                'avg_insert_ms': c.avg_insert_latency,
                'timestamps': [t.isoformat() for t in c.insert_timestamps][::step_c],
                # If we have latencies (new logs), use them. If not (legacy), use dummy 0
                'latencies': c.insert_latencies[::step_c] if c.insert_latencies else [0]*len(c.insert_timestamps[::step_c])
            })
        else:
            fog_stats.append({})
            cloud_stats.append({})
    
    # Risk level to NEWS2 score mapping
    risk_to_score = {'MINIMAL': 0, 'LOW': 1, 'MODERATE': 3, 'HIGH': 7}
    
    html_content = f'''<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VISPAC - Dashboard de Resultados</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    <style>
        :root {{
            --bg-primary: #f0f2f5;
            --bg-secondary: #ffffff;
            --bg-card: #ffffff;
            --text-primary: #333333;
            --text-secondary: #666666;
            --accent: #0077b6;
            --accent-secondary: #7209b7;
            --success: #00b4d8;
            --warning: #ff9f1c;
            --danger: #e71d36;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, var(--bg-secondary), #e0e7ff);
            border-radius: 15px;
            border: 1px solid rgba(0, 119, 182, 0.2);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            color: var(--accent);
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            color: var(--text-secondary);
        }}
        
        .dashboard {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            max-width: 1600px;
            margin: 0 auto;
        }}
        
        .card {{
            background: var(--bg-card);
            padding: 20px;
            border-radius: 15px;
            border: 1px solid rgba(0,0,0,0.1);
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            transition: transform 0.3s ease;
        }}
        
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.1);
        }}
        
        .card h2 {{
            color: var(--accent);
            margin-bottom: 15px;
            font-size: 1.3em;
            border-bottom: 1px solid rgba(0, 212, 255, 0.3);
            padding-bottom: 10px;
        }}
        
        .card.full-width {{
            grid-column: 1 / -1;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        
        .summary-item {{
            background: var(--bg-secondary);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }}
        
        .summary-item .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: var(--accent);
        }}
        
        .summary-item .label {{
            color: var(--text-secondary);
            font-size: 0.9em;
            margin-top: 5px;
        }}
        
        .chart-container {{
            position: relative;
            height: 300px;
        }}
        
        .risk-evolution {{
            max-height: 600px;
            overflow-y: auto;
        }}
        
        .patient-chart {{
            margin-bottom: 20px;
            padding: 15px;
            background: var(--bg-secondary);
            border-radius: 10px;
        }}
        
        .patient-chart h3 {{
            color: var(--text-primary);
            margin-bottom: 10px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        th {{
            color: var(--accent);
            font-weight: 600;
        }}
        
        .risk-high {{ color: #ff4444; }}
        .risk-moderate {{ color: #ffaa00; }}
        .risk-low {{ color: #00d4ff; }}
        .risk-minimal {{ color: #00ff88; }}
        
        #patientSelect {{
            width: 100%;
            padding: 10px;
            margin-bottom: 15px;
            background: var(--bg-secondary);
            color: var(--text-primary);
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 8px;
            font-size: 1em;
        }}
        
        .generated-time {{
            text-align: center;
            color: var(--text-secondary);
            margin-top: 30px;
            padding: 10px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 VISPAC Dashboard</h1>
        <p>Análise de Resultados dos Experimentos - Monitoramento de Sinais Vitais em Edge Computing</p>
    </div>
    
    <div class="dashboard">
        <!-- Summary Cards -->
        <div class="card full-width">
            <h2>📊 Resumo Comparativo</h2>
            <table>
                <thead>
                    <tr>
                        <th>Métrica</th>
                        {''.join(f'<th>{name}</th>' for name in scenario_names)}
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Transmissões</td>
                        {''.join(f'<td>{s["total_transmissions"]:,}</td>' for s in summaries)}
                    </tr>
                    <tr>
                        <td>Dados Brutos (MB)</td>
                        {''.join(f'<td>{s["total_raw_bytes"]/1024/1024:.2f}</td>' for s in summaries)}
                    </tr>
                    <tr>
                        <td>Dados Comprimidos (MB)</td>
                        {''.join(f'<td>{s["total_compressed_bytes"]/1024/1024:.2f}</td>' for s in summaries)}
                    </tr>
                    <tr>
                        <td>Compressão Total (%)</td>
                        {''.join(f'<td>{s["overall_compression"]:.1f}%</td>' for s in summaries)}
                    </tr>
                    <tr>
                        <td>PRD Média (%)</td>
                        {''.join(f'<td>{s["avg_prd"]:.4f}%</td>' for s in summaries)}
                    </tr>
                    <tr>
                        <td>Latência Média (ms)</td>
                        {''.join(f'<td>{s["avg_latency"]:.1f} ± {s["std_latency"]:.1f}</td>' for s in summaries)}
                    </tr>
                    <tr>
                        <td>CPU Média (%)</td>
                        {''.join(f'<td>{s["avg_cpu"]:.1f}%</td>' if s["avg_cpu"] > 0 else '<td>N/A</td>' for s in summaries)}
                    </tr>
                    <tr>
                        <td>Memória Média (MB)</td>
                        {''.join(f'<td>{s["avg_memory"]:.1f}</td>' if s["avg_memory"] > 0 else '<td>N/A</td>' for s in summaries)}
                    </tr>
                </tbody>
            </table>
        </div>
        
        <!-- Compression Chart -->
        <div class="card">
            <h2>📉 Taxa de Compressão</h2>
            <div class="chart-container">
                <canvas id="compressionChart"></canvas>
            </div>
        </div>
        
        <!-- Latency Chart -->
        <div class="card">
            <h2>⏱️ Latência de Transmissão</h2>
            <div class="chart-container">
                <canvas id="latencyChart"></canvas>
            </div>
        </div>
        
        <!-- PRD Chart -->
        <div class="card">
            <h2>📈 Distorção do Sinal (PRD)</h2>
            <div class="chart-container">
                <canvas id="prdChart"></canvas>
            </div>
        </div>
        
        <!-- Resource Usage Chart -->
        <div class="card">
            <h2>💻 Uso de Recursos</h2>
            <div class="chart-container">
                <canvas id="resourceChart"></canvas>
            </div>
        </div>
        
        <!-- Risk Evolution (Scenario 3 only) -->
        {'<div class="card full-width"><h2>🔄 Evolução do Risco dos Pacientes (Cenário 3)</h2><div><label for="patientSelect">Selecione o Paciente:</label><select id="patientSelect" onchange="updateRiskChart()"></select></div><div class="chart-container" style="height: 400px;"><canvas id="riskEvolutionChart"></canvas></div></div>' if risk_histories else ''}
        
        <!-- Backoff/Collection Interval Evolution (Scenario 3 only) -->
        {'<div class="card full-width"><h2>⏱️ Evolução do Intervalo de Coleta (Cenário 3)</h2><div><label for="backoffPatientSelect">Selecione o Paciente:</label><select id="backoffPatientSelect" onchange="updateBackoffChart()"></select></div><div class="chart-container" style="height: 400px;"><canvas id="backoffChart"></canvas></div></div>' if backoff_histories else ''}
        
        <!-- Fog & Cloud Metrics -->
        <div class="card full-width">
            <h2>☁️ Desempenho Fog & Cloud</h2>
            <div class="chart-container" style="height: 400px;">
                <canvas id="fogCloudChart"></canvas>
            </div>
            <div class="metrics-grid" style="margin-top: 20px;">
                <div class="metric-card">
                    <h3>Fog Process Latency</h3>
                    <p id="fogLatencyVal">-</p>
                </div>
                <div class="metric-card">
                    <h3>Fog Throughput</h3>
                    <p id="fogThroughputVal">-</p>
                </div>
                <div class="metric-card">
                    <h3>Cloud Throughput</h3>
                    <p id="cloudThroughputVal">-</p>
                </div>
                <div class="metric-card">
                    <h3>Cloud Stored</h3>
                    <p id="cloudStoredVal">-</p>
                </div>
            </div>
        </div>
    </div>
    
    <div class="generated-time">
        Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    
    <script>
        // Chart.js default configuration
        Chart.defaults.color = '#a0a0a0';
        Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.1)';
        
        const scenarioNames = {json.dumps(scenario_names)};
        const compressionData = {json.dumps(compression_data)};
        const latencyData = {json.dumps(latency_data)};
        const prdData = {json.dumps(prd_data)};
        const cpuData = {json.dumps(cpu_data)};
        const memoryData = {json.dumps(memory_data)};
        const riskHistories = {json.dumps(risk_histories)};
        const backoffHistories = {json.dumps(backoff_histories)};
        const fogStats = {json.dumps(fog_stats)};
        const cloudStats = {json.dumps(cloud_stats)};
        
        // Risk level colors
        const riskColors = {{
            'MINIMAL': '#00ff88',
            'LOW': '#00d4ff',
            'MODERATE': '#ffaa00',
            'HIGH': '#ff4444'
        }};
        
        const riskToScore = {{
            'MINIMAL': 0,
            'LOW': 1,
            'MODERATE': 3,
            'HIGH': 7
        }};
        
        // Compression Chart
        new Chart(document.getElementById('compressionChart'), {{
            type: 'bar',
            data: {{
                labels: scenarioNames,
                datasets: [{{
                    label: 'Compressão (%)',
                    data: compressionData,
                    backgroundColor: ['#00ff88', '#00d4ff', '#7b2cbf'],
                    borderRadius: 5
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100
                    }}
                }}
            }}
        }});
        
        // Latency Chart
        new Chart(document.getElementById('latencyChart'), {{
            type: 'bar',
            data: {{
                labels: scenarioNames,
                datasets: [{{
                    label: 'Latência (ms)',
                    data: latencyData,
                    backgroundColor: ['#00ff88', '#00d4ff', '#7b2cbf'],
                    borderRadius: 5
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
        
        // PRD Chart
        new Chart(document.getElementById('prdChart'), {{
            type: 'bar',
            data: {{
                labels: scenarioNames,
                datasets: [{{
                    label: 'PRD (%)',
                    data: prdData,
                    backgroundColor: ['#00ff88', '#00d4ff', '#7b2cbf'],
                    borderRadius: 5
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
        
        // Resource Chart
        new Chart(document.getElementById('resourceChart'), {{
            type: 'bar',
            data: {{
                labels: scenarioNames,
                datasets: [
                    {{
                        label: 'CPU (%)',
                        data: cpuData,
                        backgroundColor: '#ff6b6b',
                        borderRadius: 5
                    }},
                    {{
                        label: 'Memória (MB)',
                        data: memoryData,
                        backgroundColor: '#4ecdc4',
                        borderRadius: 5
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
        
        // Risk Evolution Chart (Scenario 3)
        let riskChart = null;
        
        function initRiskSelect() {{
            const select = document.getElementById('patientSelect');
            if (!select) return;
            
            const patients = Object.keys(riskHistories).sort();
            patients.forEach(p => {{
                const option = document.createElement('option');
                option.value = p;
                option.textContent = p;
                select.appendChild(option);
            }});
            
            if (patients.length > 0) {{
                updateRiskChart();
            }}
        }}
        
        function updateRiskChart() {{
            const select = document.getElementById('patientSelect');
            if (!select) return;
            
            const patientId = select.value;
            const history = riskHistories[patientId];
            if (!history) return;
            
            const ctx = document.getElementById('riskEvolutionChart');
            
            if (riskChart) {{
                riskChart.destroy();
            }}
            
            // Prepare data with risk scores
            const dataPoints = history.timestamps.map((ts, i) => ({{
                x: new Date(ts),
                y: history.news2_scores[i],
                risk: history.risks[i]
            }}));
            
            riskChart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    datasets: [{{
                        label: 'Escore NEWS2',
                        data: dataPoints,
                        borderColor: '#00d4ff',
                        backgroundColor: 'rgba(0, 212, 255, 0.1)',
                        fill: true,
                        tension: 0.1,
                        pointRadius: 6,
                        pointBackgroundColor: dataPoints.map(p => riskColors[p.risk]),
                        pointBorderColor: '#fff',
                        pointBorderWidth: 2
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        title: {{
                            display: true,
                            text: `Evolução do Risco - ${{patientId}}`,
                            color: '#333333',
                            font: {{ size: 16 }}
                        }},
                        legend: {{ display: false }},
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    const point = context.raw;
                                    return `NEWS2: ${{point.y}} (${{point.risk}})`;
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            type: 'time',
                            time: {{
                                displayFormats: {{
                                    hour: 'HH:mm'
                                }}
                            }},
                            title: {{
                                display: true,
                                text: 'Tempo da Simulação'
                            }}
                        }},
                        y: {{
                            beginAtZero: true,
                            max: 12,
                            title: {{
                                display: true,
                                text: 'Escore NEWS2'
                            }},
                            ticks: {{
                                stepSize: 1
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        // Backoff Chart Functions
        let backoffChart = null;
        
        function initBackoffSelect() {{
            const select = document.getElementById('backoffPatientSelect');
            if (!select) return;
            
            const patients = Object.keys(backoffHistories).sort();
            patients.forEach(p => {{
                const option = document.createElement('option');
                option.value = p;
                option.textContent = p;
                select.appendChild(option);
            }});
            
            if (patients.length > 0) {{
                updateBackoffChart();
            }}
        }}
        
        function updateBackoffChart() {{
            const select = document.getElementById('backoffPatientSelect');
            if (!select) return;
            
            const patientId = select.value;
            const history = backoffHistories[patientId];
            if (!history) return;
            
            const ctx = document.getElementById('backoffChart');
            
            if (backoffChart) {{
                backoffChart.destroy();
            }}
            
            // Prepare data - show new interval over time, color by signal type
            const dataPoints = history.timestamps.map((ts, i) => ({{
                x: new Date(ts),
                y: history.new_intervals[i],
                signal: history.signals[i],
                oldInterval: history.old_intervals[i]
            }}));
            
            backoffChart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    datasets: [{{
                        label: 'Intervalo de Coleta (s)',
                        data: dataPoints,
                        borderColor: '#00d4ff',
                        backgroundColor: 'rgba(0, 212, 255, 0.1)',
                        fill: true,
                        tension: 0.3,
                        stepped: 'before',
                        pointRadius: 6,
                        pointBackgroundColor: dataPoints.map(p => p.signal === 'HR' ? '#ff6b6b' : '#4ecdc4'),
                        pointBorderColor: '#fff',
                        pointBorderWidth: 2
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        title: {{
                            display: true,
                            text: `Evolução do Intervalo de Coleta - ${{patientId}}`,
                            color: '#333333',
                            font: {{ size: 16 }}
                        }},
                        legend: {{ display: false }},
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    const point = context.raw;
                                    return `${{point.signal}}: ${{point.oldInterval}}s → ${{point.y}}s`;
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            type: 'time',
                            time: {{
                                displayFormats: {{
                                    hour: 'HH:mm'
                                }}
                            }},
                            title: {{
                                display: true,
                                text: 'Tempo da Simulação'
                            }}
                        }},
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Intervalo de Coleta (segundos)'
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        // Initialize
        initRiskSelect();
        initBackoffSelect();
        initFogCloudChart();
        
        function initFogCloudChart() {{
            const ctx = document.getElementById('fogCloudChart');
            if (!ctx) return;
            
            // Find vispac scenario data (usually index 2, but check)
            const idx = scenarios.findIndex(s => s.includes('vispac'));
            if (idx === -1) return;
            
            const fog = fogStats[idx];
            const cloud = cloudStats[idx];
            
            // Update summary cards
            document.getElementById('fogLatencyVal').textContent = fog.avg_process_ms > 0 ? fog.avg_process_ms.toFixed(2) + ' ms' : 'N/A';
            document.getElementById('fogThroughputVal').textContent = fog.total_batches + ' batches';
            document.getElementById('cloudStoredVal').textContent = cloud.total_items + ' items';
            
            // Determine start/end times to calculate rough throughput
            if (fog.timestamps && fog.timestamps.length > 1) {{
                 const start = new Date(fog.timestamps[0]);
                 const end = new Date(fog.timestamps[fog.timestamps.length-1]);
                 const seconds = (end - start) / 1000;
                 if (seconds > 0) {{
                     const tput = fog.total_batches / seconds;
                     document.getElementById('fogThroughputVal').textContent += ` (${{tput.toFixed(2)}}/s)`;
                 }}
            }}
            
            // Prepare datasets
            const datasets = [];
            
            // Fog Processing Latency (if available)
            if (fog.timestamps && fog.latencies && fog.latencies.some(l => l > 0)) {{
                datasets.push({{
                    label: 'Fog: Latência de Processamento (ms)',
                    data: fog.timestamps.map((t, i) => ({{ x: t, y: fog.latencies[i] }})),
                    borderColor: '#ffaa00',
                    backgroundColor: 'rgba(255, 170, 0, 0.1)',
                    yAxisID: 'yLatency',
                    pointRadius: 2
                }});
            }}
            
            // Cloud Items Stored (Cumulative)
            // If we have timestamps for cloud, show accumulation
            if (cloud.timestamps && cloud.timestamps.length > 0) {{
                // Create cumulative count
                let count = 0;
                const cumulative = cloud.timestamps.map((t, i) => {{
                    // Estimate items per batch mostly 1 for legacy, or look at logic
                    // Cloud legacy logs: "Stored X items". We summed total.
                    // For chart, let's assume linear accumulation or just plot events
                    // Actually, if we scraped "stored X items", we can sum X.
                    // Ideally we parsed X per line. LogParser stores total.
                    // But we captured timestamps. Let's assume each timestamp = 1 event (batch).
                    // CloudMetrics.total_items_stored is sum of items.
                    // But our parser logic: `self.cloud_metrics.insert_latencies.append` happens once per log line.
                    // So timestamps match log lines.
                    count += (cloud.total_items / cloud.timestamps.length); // rough approx per line
                    return {{ x: t, y: count }};
                }});
                
                datasets.push({{
                    label: 'Cloud: Itens Armazenados (Acumulado)',
                    data: cumulative,
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    yAxisID: 'yCount',
                    fill: true,
                    pointRadius: 0
                }});
            }}

            new Chart(ctx, {{
                type: 'line',
                data: {{ datasets: datasets }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {{
                        mode: 'index',
                        intersect: false,
                    }},
                    scales: {{
                        x: {{ type: 'time', time: {{ unit: 'minute' }} }},
                        yLatency: {{
                            type: 'linear',
                            display: datasets.some(d => d.yAxisID === 'yLatency'),
                            position: 'left',
                            title: {{ display: true, text: 'Latência (ms)' }}
                        }},
                        yCount: {{
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {{ display: true, text: 'Total Itens' }},
                            grid: {{ drawOnChartArea: false }}
                        }}
                    }}
                }}
            }});
        }}
    </script>
</body>
</html>'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML Dashboard generated: {output_path}")


def generate_comparison_report(summaries: List[Dict], output_path: Path) -> None:
    """Generate markdown comparison report."""
    with open(output_path, 'w') as f:
        f.write("# VISPAC Experiment Results - Comparison Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary Table\n\n")
        f.write("| Metric | " + " | ".join(s['scenario'] for s in summaries) + " |\n")
        f.write("|--------|" + "|".join(["--------"] * len(summaries)) + "|\n")
        
        rows = [
            ('Edges', 'num_edges', '{:.0f}'),
            ('Transmissions', 'total_transmissions', '{:.0f}'),
            ('Raw Data (MB)', 'total_raw_bytes', lambda x: f"{x/1024/1024:.2f}"),
            ('Compressed (MB)', 'total_compressed_bytes', lambda x: f"{x/1024/1024:.2f}"),
            ('Compression (%)', 'avg_compression', '{:.1f} ± {std_compression:.1f}'),
            ('PRD (%)', 'avg_prd', '{:.4f} ± {std_prd:.4f}'),
            ('Latency (ms)', 'avg_latency', '{:.1f} ± {std_latency:.1f}'),
            ('CPU (%)', 'avg_cpu', '{:.1f}'),
            ('Memory (MB)', 'avg_memory', '{:.1f}'),
        ]
        
        for label, key, fmt in rows:
            values = []
            for s in summaries:
                if callable(fmt):
                    values.append(fmt(s.get(key, 0)))
                elif '{std_' in fmt:
                    std_key = key.replace('avg_', 'std_')
                    values.append(fmt.format(s.get(key, 0), **{f'std_{key.split("_")[1]}': s.get(std_key, 0)}))
                else:
                    values.append(fmt.format(s.get(key, 0)))
            f.write(f"| {label} | " + " | ".join(values) + " |\n")
        
        f.write("\n## Key Findings\n\n")
        
        # Compare scenarios
        if len(summaries) >= 3:
            baseline = next((s for s in summaries if 'baseline' in s['scenario']), None)
            vispac = next((s for s in summaries if 'vispac' in s['scenario']), None)
            
            if baseline and vispac:
                if baseline['total_raw_bytes'] > 0 and vispac['total_compressed_bytes'] > 0:
                    reduction = (1 - vispac['total_compressed_bytes'] / baseline['total_raw_bytes']) * 100
                    f.write(f"- **Data Reduction (ViSPAC vs Baseline):** {reduction:.1f}%\n")
                
                if vispac['avg_prd'] > 0:
                    f.write(f"- **Signal Distortion (PRD):** {vispac['avg_prd']:.4f}%\n")
    
    print(f"Report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze VISPAC experiment logs')
    parser.add_argument('logs_dir', type=Path, help='Directory containing log files')
    parser.add_argument('--output', '-o', type=Path, default=Path('results'),
                       help='Output directory for results')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all scenario subdirectories')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    if args.compare:
        # Compare all scenarios in subdirectories
        summaries = []
        parsers = {}
        for scenario_dir in sorted(args.logs_dir.iterdir()):
            if scenario_dir.is_dir() and scenario_dir.name.startswith('scenario'):
                print(f"\n{'='*60}")
                print(f"Processing: {scenario_dir.name}")
                print('='*60)
                
                log_parser = LogParser(scenario_dir.name)
                log_parser.parse_directory(scenario_dir)
                parsers[scenario_dir.name] = log_parser
                
                # Export CSV
                csv_path = args.output / f"{scenario_dir.name}_metrics.csv"
                export_csv(log_parser.metrics, csv_path)
                
                summaries.append(log_parser.get_summary())
        
        # Generate outputs
        if summaries:
            report_path = args.output / "comparison_report.md"
            generate_comparison_report(summaries, report_path)
            
            html_path = args.output / "dashboard.html"
            generate_html_dashboard(summaries, parsers, html_path)
    else:
        # Single scenario
        scenario_name = args.logs_dir.name
        print(f"\nProcessing: {scenario_name}")
        
        log_parser = LogParser(scenario_name)
        log_parser.parse_directory(args.logs_dir)
        
        # Export CSV
        csv_path = args.output / f"{scenario_name}_metrics.csv"
        export_csv(log_parser.metrics, csv_path)
        
        # Generate HTML for single scenario
        summary = log_parser.get_summary()
        html_path = args.output / f"{scenario_name}_dashboard.html"
        generate_html_dashboard([summary], {scenario_name: log_parser}, html_path)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"SUMMARY: {scenario_name}")
        print('='*60)
        print(f"Edges: {summary['num_edges']}")
        print(f"Transmissions: {summary['total_transmissions']}")
        print(f"Raw Data: {summary['total_raw_bytes']/1024/1024:.2f} MB")
        print(f"Compressed: {summary['total_compressed_bytes']/1024/1024:.2f} MB")
        print(f"Compression: {summary['avg_compression']:.1f}% ± {summary['std_compression']:.1f}%")
        print(f"PRD: {summary['avg_prd']:.4f}% ± {summary['std_prd']:.4f}%")
        print(f"Latency: {summary['avg_latency']:.1f}ms ± {summary['std_latency']:.1f}ms")
        if summary['avg_cpu'] > 0:
            print(f"CPU: {summary['avg_cpu']:.1f}% (max: {summary['max_cpu']:.1f}%)")
        if summary['avg_memory'] > 0:
            print(f"Memory: {summary['avg_memory']:.1f}MB (max: {summary['max_memory']:.1f}MB)")


if __name__ == '__main__':
    main()
