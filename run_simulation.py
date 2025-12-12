#!/usr/bin/env python3
"""
Interactive script to run VISPAC simulation with different datasets
"""

import os
import sys
import subprocess

def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')

def print_header():
    print("=" * 60)
    print("    VISPAC Edge Prototype - Dataset Selector")
    print("=" * 60)
    print()

def print_dataset_info(dataset_type):
    """Shows information about the dataset"""
    import pandas as pd
    
    dataset_paths = {
        "low_risk": "datasets/low_risk/low_risk_processed.csv",
        "high_risk": "datasets/high_risk/high_risk_processed.csv",
    }
    
    if dataset_type not in dataset_paths:
        return
    
    path = dataset_paths[dataset_type]
    
    if not os.path.exists(path):
        print(f"⚠️  Dataset not found: {path}")
        return
    
    df = pd.read_csv(path)
    
    print(f"\n📊 Dataset Information: {dataset_type.upper()}")
    print("-" * 60)
    print(f"File: {path}")
    print(f"Samples: {len(df)}")
    print(f"\nStatistics:")
    print(f"  HR (bpm):")
    print(f"    Mean: {df['hr'].mean():.1f}")
    print(f"    Range: {df['hr'].min()}-{df['hr'].max()}")
    print(f"    StdDev: ±{df['hr'].std():.1f}")
    print(f"  SpO2 (%):")
    print(f"    Mean: {df['spo2'].mean():.1f}")
    print(f"    Range: {df['spo2'].min()}-{df['spo2'].max()}")
    print(f"    StdDev: ±{df['spo2'].std():.1f}")
    print()

def show_menu():
    print("\n📋 Choose simulation type:")
    print()
    print("1) 🟢 LOW RISK     - Stable patients (HR: 60-100, SpO2: 95-100)")
    print("2) 🔴 HIGH RISK    - Critical patients (HR: 85-160, SpO2: 75-95)")
    print("3) ℹ️  View dataset statistics")
    print("4) 🔧 Advanced settings")
    print("5) ❌ Exit")
    print()

def show_advanced_config():
    clear_screen()
    print_header()
    print("🔧 Advanced Settings")
    print("-" * 60)
    print()
    print("Current environment variables:")
    print(f"  DATASET_TYPE: {os.environ.get('DATASET_TYPE', 'not set')}")
    print(f"  API_URL: {os.environ.get('API_URL', 'http://127.0.0.1:8000/vispac/upload_batch')}")
    print(f"  EDGE_USE_MQTT: {os.environ.get('EDGE_USE_MQTT', '0')}")
    
    if os.environ.get('EDGE_USE_MQTT') == '1':
        print(f"  MQTT_BROKER: {os.environ.get('MQTT_BROKER', '127.0.0.1')}")
        print(f"  MQTT_PORT: {os.environ.get('MQTT_PORT', '1883')}")
    
    print()
    print("To modify, use 'export VARIABLE=value' before running")
    print()
    input("Press ENTER to go back...")

def show_stats():
    clear_screen()
    print_header()
    print("📊 Dataset Statistics")
    print("=" * 60)
    
    print_dataset_info("low_risk")
    print_dataset_info("high_risk")
    
    input("\nPress ENTER to go back...")

def run_simulation(dataset_type):
    clear_screen()
    print_header()
    
    # Set dataset type
    os.environ['DATASET_TYPE'] = dataset_type
    
    dataset_paths = {
        "low_risk": "datasets/low_risk/low_risk_processed.csv",
        "high_risk": "datasets/high_risk/high_risk_processed.csv",
    }
    
    # Check if dataset exists
    if dataset_type in dataset_paths:
        dataset_path = dataset_paths[dataset_type]
        if not os.path.exists(dataset_path):
            print(f"⚠️  Dataset not found: {dataset_path}")
            print()
            print("Run first: python download_datasets.py")
            print()
            input("Press ENTER to go back...")
            return
    
    print(f"🚀 Starting simulation with dataset: {dataset_type.upper()}")
    print()
    print_dataset_info(dataset_type)
    
    print("=" * 60)
    print("⚠️  Press Ctrl+C to stop the simulation")
    print("=" * 60)
    print()
    
    input("Press ENTER to start...")
    
    # Run the prototype
    try:
        # Use venv python if available
        python_cmd = '.venv/bin/python' if os.path.exists('.venv/bin/python') else 'python'
        subprocess.run([python_cmd, 'vispac_edge_prototype.py'])
    except KeyboardInterrupt:
        print("\n\n✋ Simulation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error running simulation: {e}")
    
    print()
    input("Press ENTER to go back to menu...")

def main():
    while True:
        clear_screen()
        print_header()
        show_menu()
        
        try:
            choice = input("Enter your choice [1-5]: ").strip()
            
            if choice == '1':
                run_simulation("low_risk")
            elif choice == '2':
                run_simulation("high_risk")
            elif choice == '3':
                show_stats()
            elif choice == '4':
                show_advanced_config()
            elif choice == '5':
                clear_screen()
                print("👋 Goodbye!")
                sys.exit(0)
            else:
                print("\n❌ Invalid option!")
                input("Press ENTER to continue...")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("Press ENTER to continue...")

if __name__ == '__main__':
    main()
