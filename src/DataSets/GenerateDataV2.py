"""
Production-Grade Latency Dataset Generator v2
==============================================
Generates 90 days of synthetic monitoring data with:
  - Realistic traffic patterns (hourly/daily/weekly cycles)
  - 6 distinct failure patterns repeated 2x each (12 total incidents)
  - Strong causal precursor signals (CPU/memory/DB deterioration before latency)
  - Better feature engineering (rates of change, anomaly flags, interactions)
  - Documented failure windows for model validation

Each failure type gets 2-3 instances at different intensities/times to give
the model actual learning examples.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import percentileofscore

np.random.seed(42)

# ======================================================================
# CONFIGURATION
# ======================================================================
START_DATE = datetime(2026, 4, 1)
DAYS = 90
INTERVAL_MINUTES = 5
TOTAL_POINTS = (DAYS * 24 * 60) // INTERVAL_MINUTES
MAX_TRAFFIC = 800

print(f"Generating {DAYS} days of data ({TOTAL_POINTS} samples)")

# ======================================================================
# 1. TIMESTAMPS & BASE TRAFFIC
# ======================================================================
timestamps = [START_DATE + timedelta(minutes=i * INTERVAL_MINUTES) for i in range(TOTAL_POINTS)]

def generate_ar1_noise(length, rho=0.8, sigma=1.0):
    """AR(1) noise for realistic autocorrelation."""
    noise = np.zeros(length)
    noise[0] = np.random.normal(0, sigma)
    for i in range(1, length):
        noise[i] = rho * noise[i-1] + np.random.normal(0, sigma)
    return noise

# Traffic patterns
traffic_noise = generate_ar1_noise(TOTAL_POINTS, rho=0.85, sigma=8)
request_count = []

for i, ts in enumerate(timestamps):
    hour = ts.hour
    dow = ts.weekday()
    
    # Hourly pattern
    if 8 <= hour < 12:
        base = np.random.uniform(350, 400)
    elif 12 <= hour < 18:
        base = np.random.uniform(450, 550)
    elif 18 <= hour < 22:
        base = np.random.uniform(300, 350)
    else:
        base = np.random.uniform(100, 150)
    
    # Weekend reduction
    if dow >= 5:
        base *= 0.7
    
    request_count.append(base + traffic_noise[i])

request_count = np.clip(np.array(request_count), 10, None)

# ======================================================================
# 2. BASELINE METRICS (CPU, Memory, DB latency)
# ======================================================================
active_users = request_count * 0.45 + generate_ar1_noise(TOTAL_POINTS, rho=0.9, sigma=2)
active_users = np.clip(active_users, 5, None)

cpu_usage = (
    25 
    + (request_count / MAX_TRAFFIC) * 45 
    + generate_ar1_noise(TOTAL_POINTS, rho=0.7, sigma=1.5)
)
cpu_usage = np.clip(cpu_usage, 5, 75)  # Keep baseline low to show anomalies

memory_usage = (
    35 
    + (request_count / MAX_TRAFFIC) * 35 
    + generate_ar1_noise(TOTAL_POINTS, rho=0.8, sigma=1)
)
memory_usage = np.clip(memory_usage, 15, 70)

db_query_time = (
    8
    + (cpu_usage / 100) * 12
    + (memory_usage / 100) * 8
    + generate_ar1_noise(TOTAL_POINTS, rho=0.8, sigma=0.5)
)
db_query_time = np.clip(db_query_time, 5, 40)

# ======================================================================
# 3. INJECT FAILURE PATTERNS (Repeated 2x each for learning)
# ======================================================================

failure_windows = []  # Track for documentation

def inject_failure(name, start_day, duration_hours, cpu_mult=1.0, mem_delta=0, 
                   db_mult=1.0, impact_on_latency=1.0):
    """
    Inject a failure pattern into the metrics.
    
    Args:
      start_day: Day (0-89) when failure starts
      duration_hours: How long the failure lasts
      cpu_mult: CPU multiplier (1.5 = 50% increase)
      mem_delta: Memory additive change (e.g., +20 means +20%)
      db_mult: Database query time multiplier
      impact_on_latency: Later used to scale latency impact
    """
    start_idx = int(start_day * 24 * 60 / INTERVAL_MINUTES)
    end_idx = int((start_day + duration_hours/24) * 24 * 60 / INTERVAL_MINUTES)
    
    for i in range(start_idx, min(end_idx, TOTAL_POINTS)):
        # Ramp up and down for realism
        progress = (i - start_idx) / (end_idx - start_idx)
        ramp = np.sin(np.pi * progress) ** 0.5  # Ease in/out
        
        cpu_usage[i] = np.clip(cpu_usage[i] * (1 + (cpu_mult - 1) * ramp), 5, 99)
        memory_usage[i] = np.clip(memory_usage[i] + mem_delta * ramp, 15, 95)
        db_query_time[i] *= (1 + (db_mult - 1) * ramp)
    
    failure_windows.append({
        'name': name,
        'start_day': start_day,
        'duration_hours': duration_hours,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'latency_impact': impact_on_latency
    })
    print(f"  Injected {name:25s} @ day {start_day:2.1f} ({duration_hours:2.0f}h), CPU×{cpu_mult:.1f}, Mem+{mem_delta:+3.0f}%, DB×{db_mult:.1f}, Latency×{impact_on_latency:.1f}")


print("\nInjecting failure patterns:")

# Memory Leak (progressive increase) - 2 instances
inject_failure("Memory Leak #1", start_day=8, duration_hours=18, 
               cpu_mult=1.2, mem_delta=30, db_mult=1.3, impact_on_latency=0.8)
inject_failure("Memory Leak #2", start_day=45, duration_hours=16, 
               cpu_mult=1.15, mem_delta=25, db_mult=1.25, impact_on_latency=0.75)

# CPU Spike (sudden, hard on latency) - 2 instances
inject_failure("CPU Spike #1", start_day=15, duration_hours=6, 
               cpu_mult=2.2, mem_delta=10, db_mult=1.8, impact_on_latency=1.8)
inject_failure("CPU Spike #2", start_day=62, duration_hours=5, 
               cpu_mult=2.0, mem_delta=8, db_mult=1.6, impact_on_latency=1.6)

# Database Lock (slow query degradation) - 2 instances
inject_failure("Database Lock #1", start_day=24, duration_hours=12, 
               cpu_mult=1.4, mem_delta=15, db_mult=3.0, impact_on_latency=1.5)
inject_failure("Database Lock #2", start_day=68, duration_hours=10, 
               cpu_mult=1.3, mem_delta=12, db_mult=2.8, impact_on_latency=1.4)

# Request Flood (sustained high load) - 2 instances
# This one modifies request_count instead
print("  Injected Request Flood #1            @ day  32 ( 8h), Req×1.8")
flood_start_1 = int(32 * 24 * 60 / INTERVAL_MINUTES)
flood_end_1 = int((32 + 8/24) * 24 * 60 / INTERVAL_MINUTES)
for i in range(flood_start_1, min(flood_end_1, TOTAL_POINTS)):
    progress = (i - flood_start_1) / (flood_end_1 - flood_start_1)
    ramp = np.sin(np.pi * progress) ** 0.5
    request_count[i] *= (1 + 0.8 * ramp)
    cpu_usage[i] = np.clip(cpu_usage[i] * (1 + 0.5 * ramp), 5, 99)
    memory_usage[i] = np.clip(memory_usage[i] + 10 * ramp, 15, 95)
failure_windows.append({
    'name': 'Request Flood #1', 'start_day': 32, 'duration_hours': 8,
    'start_idx': flood_start_1, 'end_idx': flood_end_1, 'latency_impact': 0.6
})

print("  Injected Request Flood #2            @ day  75 ( 7h), Req×1.7")
flood_start_2 = int(75 * 24 * 60 / INTERVAL_MINUTES)
flood_end_2 = int((75 + 7/24) * 24 * 60 / INTERVAL_MINUTES)
for i in range(flood_start_2, min(flood_end_2, TOTAL_POINTS)):
    progress = (i - flood_start_2) / (flood_end_2 - flood_start_2)
    ramp = np.sin(np.pi * progress) ** 0.5
    request_count[i] *= (1 + 0.7 * ramp)
    cpu_usage[i] = np.clip(cpu_usage[i] * (1 + 0.45 * ramp), 5, 99)
    memory_usage[i] = np.clip(memory_usage[i] + 8 * ramp, 15, 95)
failure_windows.append({
    'name': 'Request Flood #2', 'start_day': 75, 'duration_hours': 7,
    'start_idx': flood_start_2, 'end_idx': flood_end_2, 'latency_impact': 0.55
})

# Cascading Failure (multiple systems fail together) - 2 instances
inject_failure("Cascading Failure #1", start_day=38, duration_hours=9, 
               cpu_mult=1.8, mem_delta=35, db_mult=2.5, impact_on_latency=2.0)
inject_failure("Cascading Failure #2", start_day=80, duration_hours=8, 
               cpu_mult=1.75, mem_delta=32, db_mult=2.4, impact_on_latency=1.9)

# Thermal Throttle (gradual degradation) - 2 instances
inject_failure("Thermal Throttle #1", start_day=52, duration_hours=11, 
               cpu_mult=1.3, mem_delta=5, db_mult=1.4, impact_on_latency=0.85)
inject_failure("Thermal Throttle #2", start_day=85, duration_hours=10, 
               cpu_mult=1.25, mem_delta=4, db_mult=1.35, impact_on_latency=0.8)

request_count = np.clip(request_count, 10, None)
cpu_usage = np.clip(cpu_usage, 5, 99.9)
memory_usage = np.clip(memory_usage, 10, 99.9)
db_query_time = np.clip(db_query_time, 5, 80)

# ======================================================================
# 4. CALCULATE LATENCY (Regression target)
# ======================================================================
# Latency depends on: baseline + load effect + cpu overhead + db delay + failures
p50_latency = []
latency_noise = generate_ar1_noise(TOTAL_POINTS, rho=0.6, sigma=2)

for i, ts in enumerate(timestamps):
    hour_fraction = (ts.hour + ts.minute / 60) / 24
    daily_cycle = 8 * np.sin(2 * np.pi * hour_fraction)  # Lighter daily cycle
    
    latency = (
        35  # Base latency
        + (request_count[i] / MAX_TRAFFIC) * 40  # Load component
        + daily_cycle
        + (cpu_usage[i] ** 1.1) * 0.18  # Non-linear CPU effect
        + db_query_time[i] * 1.5  # DB latency multiplier
        + latency_noise[i]
    )
    
    # Apply failure-specific latency impacts
    for fw in failure_windows:
        if fw['start_idx'] <= i < fw['end_idx']:
            progress = (i - fw['start_idx']) / (fw['end_idx'] - fw['start_idx'])
            impact = np.sin(np.pi * progress) ** 0.5 * fw['latency_impact']
            latency += 150 * impact  # Strong latency spike during failure
    
    p50_latency.append(latency)

p50_latency = np.array(p50_latency)
p50_latency = np.clip(p50_latency, 20, 800)

# Percentile latencies
p95_latency = p50_latency * (1.4 + (cpu_usage / 100) * 1.0 + generate_ar1_noise(TOTAL_POINTS, rho=0.5, sigma=0.1))
p99_latency = p50_latency * (1.8 + (cpu_usage / 100) * 2.0 + generate_ar1_noise(TOTAL_POINTS, rho=0.5, sigma=0.2))

# ======================================================================
# 5. ERROR RATE (Increases during failures)
# ======================================================================
cpu_risk = np.maximum(0, (cpu_usage - 75) / 25)
db_risk = np.maximum(0, (db_query_time - 30) / 30)
latency_risk = np.maximum(0, (p99_latency - 300) / 400)

error_noise = np.abs(generate_ar1_noise(TOTAL_POINTS, rho=0.7, sigma=0.001))
error_rate = (
    0.0008  # Very low baseline
    + (cpu_risk * 0.035)
    + (db_risk * 0.04)
    + (latency_risk * 0.05)
    + error_noise
)
error_rate = np.clip(error_rate, 0.0005, 0.15)

# ======================================================================
# 6. ADVANCED FEATURE ENGINEERING
# ======================================================================
# Rate of change (detecting rapid deterioration)
cpu_change_5m = np.gradient(cpu_usage, edge_order=2)
memory_change_5m = np.gradient(memory_usage, edge_order=2)
db_change_5m = np.gradient(db_query_time, edge_order=2)

# Rolling statistics (anomaly detection)
cpu_rolling_mean = pd.Series(cpu_usage).rolling(window=24).mean().values  # 2 hours
cpu_rolling_std = pd.Series(cpu_usage).rolling(window=24).std().values
memory_rolling_mean = pd.Series(memory_usage).rolling(window=24).mean().values
memory_rolling_std = pd.Series(memory_usage).rolling(window=24).std().values

# Replace NaNs from rolling
cpu_rolling_mean = np.nan_to_num(cpu_rolling_mean, nan=np.mean(cpu_usage))
cpu_rolling_std = np.nan_to_num(cpu_rolling_std, nan=np.std(cpu_usage))
memory_rolling_mean = np.nan_to_num(memory_rolling_mean, nan=np.mean(memory_usage))
memory_rolling_std = np.nan_to_num(memory_rolling_std, nan=np.std(memory_usage))

# Percentile anomaly flags
cpu_percentile = np.array([percentileofscore(cpu_usage, x) for x in cpu_usage])
memory_percentile = np.array([percentileofscore(memory_usage, x) for x in memory_usage])

# Interaction terms (capture combined stress)
cpu_memory_interaction = (cpu_usage / 100) * (memory_usage / 100)
cpu_load_interaction = (cpu_usage / 100) * (request_count / MAX_TRAFFIC)

# ======================================================================
# 7. CREATE DATAFRAME
# ======================================================================
df = pd.DataFrame({
    'timestamp': timestamps,
    'request_count': np.round(request_count).astype(int),
    'active_users': np.round(active_users).astype(int),
    'cpu_usage': np.round(cpu_usage, 2),
    'memory_usage': np.round(memory_usage, 2),
    'db_query_time_ms': np.round(db_query_time, 2),
    'p50_latency_ms': np.round(p50_latency, 2),
    'p95_latency_ms': np.round(p95_latency, 2),
    'p99_latency_ms': np.round(p99_latency, 2),
    'error_rate': np.round(error_rate, 5),
    
    # Advanced features
    'cpu_change_5m': np.round(cpu_change_5m, 3),
    'memory_change_5m': np.round(memory_change_5m, 3),
    'db_change_5m': np.round(db_change_5m, 3),
    'cpu_2h_mean': np.round(cpu_rolling_mean, 2),
    'cpu_2h_std': np.round(cpu_rolling_std, 2),
    'memory_2h_mean': np.round(memory_rolling_mean, 2),
    'memory_2h_std': np.round(memory_rolling_std, 2),
    'cpu_percentile': np.round(cpu_percentile, 1),
    'memory_percentile': np.round(memory_percentile, 1),
    'cpu_memory_stress': np.round(cpu_memory_interaction, 4),
    'cpu_load_stress': np.round(cpu_load_interaction, 4),
})

df.to_csv('latency_data_v2.csv', index=False)

# ======================================================================
# 8. DOCUMENT FAILURE WINDOWS (for validation)
# ======================================================================
failure_df = pd.DataFrame(failure_windows)
failure_df.to_csv('failure_windows_v2.csv', index=False)

print("\n" + "=" * 70)
print(f"✓ Generated latency_data_v2.csv ({len(df)} samples)")
print(f"✓ Generated failure_windows_v2.csv (for test set validation)")
print("=" * 70)
print(f"\nDataset summary:")
print(f"  Duration: {DAYS} days")
print(f"  p99_latency_ms: mean={df['p99_latency_ms'].mean():.1f}ms, "
      f"std={df['p99_latency_ms'].std():.1f}ms, "
      f"min={df['p99_latency_ms'].min():.1f}ms, "
      f"max={df['p99_latency_ms'].max():.1f}ms")
print(f"  Failure instances: {len(failure_windows)}")
print(f"  Features (including engineered): {len(df.columns)}")
print(f"\nFailure windows for validation:")
for fw in failure_windows:
    print(f"  {fw['name']:25s} day {fw['start_day']:5.1f} ({fw['duration_hours']:4.1f}h)")