#!/usr/bin/env python3
"""
VISPAC Log Analysis Script

Parses experiment logs from edge devices and extracts metrics for comparison
across the 3 test scenarios (baseline, static, vispac).

Usage:
    python analyze_logs.py logs/scenario1_baseline --output results/
    python analyze_logs.py --compare logs/ --output results/

Author: VISPAC Research
"""

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


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


class LogParser:
    """Parses VISPAC edge log files and extracts metrics."""
    
    # Regex patterns for log parsing
    PATTERNS = {
        # [HR COLLECT] edge-01 HIGH | 120→45 pts | 2400b→900b (62.5%)
        'collect': re.compile(
            r'\[(HR|SpO2) COLLECT\]\s+(\S+)\s+(\w+)\s+\|.*?\|.*?(\d+)b→(\d+)b\s+\((\d+\.?\d*)%\)'
        ),
        # [HR COLLECT] edge-01 HIGH | 120 pts | 2400b (RAW - no compression)
        'collect_raw': re.compile(
            r'\[(HR|SpO2) COLLECT\]\s+(\S+)\s+(\w+)\s+\|.*?\|.*?(\d+)b\s+\(RAW'
        ),
        # [HR DISTORTION] edge-01 PRD=0.0234%
        'prd': re.compile(
            r'\[(HR|SpO2) DISTORTION\]\s+(\S+)\s+PRD=(\d+\.?\d*)%'
        ),
        # [SEND] edge-01 | Batch 'HIGH' | 5 pkts | ... | 12.5ms
        'send': re.compile(
            r'\[SEND\]\s+(\S+)\s+\|.*?(\d+\.?\d*)ms'
        ),
        # [RISK] PID-001: LOW → HIGH (NEWS2=8)
        'risk': re.compile(
            r'\[RISK\]\s+(\S+):\s+\w+\s+→\s+(\w+)'
        ),
        # Timestamp at start of line
        'timestamp': re.compile(
            r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d+)'
        ),
    }
    
    def __init__(self, scenario: str):
        self.scenario = scenario
        self.metrics: Dict[str, EdgeMetrics] = {}
    
    def parse_file(self, filepath: Path) -> None:
        """Parse a single log file and extract metrics."""
        # Extract edge_id from filename (e.g., edge-01_service.log)
        filename = filepath.stem
        parts = filename.split('_')
        if len(parts) >= 2:
            edge_id = parts[0]
        else:
            edge_id = filename
        
        if edge_id not in self.metrics:
            self.metrics[edge_id] = EdgeMetrics(edge_id=edge_id, scenario=self.scenario)
        
        metrics = self.metrics[edge_id]
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    self._parse_line(line, metrics)
        except Exception as e:
            print(f"Warning: Error parsing {filepath}: {e}")
    
    def _parse_line(self, line: str, metrics: EdgeMetrics) -> None:
        """Parse a single log line and update metrics."""
        
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
        
        # Try to match send latency
        match = self.PATTERNS['send'].search(line)
        if match:
            edge_id, latency = match.groups()
            metrics.latencies.append(float(latency))
            return
    
    def parse_directory(self, directory: Path) -> None:
        """Parse all log files in a directory."""
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        log_files = list(directory.glob('*_service.log')) + list(directory.glob('*.log'))
        log_files = [f for f in log_files if '_error' not in f.name]
        
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
        total_raw = 0
        total_compressed = 0
        total_transmissions = 0
        
        for m in self.metrics.values():
            all_compression.extend(m.compression_ratios)
            all_prd.extend(m.prd_values)
            all_latency.extend(m.latencies)
            total_raw += m.total_raw_bytes
            total_compressed += m.total_compressed_bytes
            total_transmissions += m.total_transmissions
        
        def safe_avg(lst):
            return sum(lst) / len(lst) if lst else 0.0
        
        def safe_std(lst):
            if len(lst) < 2:
                return 0.0
            avg = safe_avg(lst)
            variance = sum((x - avg) ** 2 for x in lst) / len(lst)
            return variance ** 0.5
        
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
            'min_latency': min(all_latency) if all_latency else 0,
            'max_latency': max(all_latency) if all_latency else 0,
        }


def export_csv(metrics: Dict[str, EdgeMetrics], output_path: Path) -> None:
    """Export metrics to CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'edge_id', 'scenario', 'total_transmissions',
            'total_raw_bytes', 'total_compressed_bytes',
            'avg_compression_%', 'avg_prd_%', 
            'avg_latency_ms', 'std_latency_ms',
            'risk_high', 'risk_moderate', 'risk_low', 'risk_minimal'
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
                m.risk_distribution.get('HIGH', 0),
                m.risk_distribution.get('MODERATE', 0),
                m.risk_distribution.get('LOW', 0),
                m.risk_distribution.get('MINIMAL', 0),
            ])
    
    print(f"CSV exported to: {output_path}")


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
        for scenario_dir in sorted(args.logs_dir.iterdir()):
            if scenario_dir.is_dir() and scenario_dir.name.startswith('scenario'):
                print(f"\n{'='*60}")
                print(f"Processing: {scenario_dir.name}")
                print('='*60)
                
                log_parser = LogParser(scenario_dir.name)
                log_parser.parse_directory(scenario_dir)
                
                # Export CSV
                csv_path = args.output / f"{scenario_dir.name}_metrics.csv"
                export_csv(log_parser.metrics, csv_path)
                
                summaries.append(log_parser.get_summary())
        
        # Generate comparison report
        if summaries:
            report_path = args.output / "comparison_report.md"
            generate_comparison_report(summaries, report_path)
    else:
        # Single scenario
        scenario_name = args.logs_dir.name
        print(f"\nProcessing: {scenario_name}")
        
        log_parser = LogParser(scenario_name)
        log_parser.parse_directory(args.logs_dir)
        
        # Export CSV
        csv_path = args.output / f"{scenario_name}_metrics.csv"
        export_csv(log_parser.metrics, csv_path)
        
        # Print summary
        summary = log_parser.get_summary()
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


if __name__ == '__main__':
    main()
