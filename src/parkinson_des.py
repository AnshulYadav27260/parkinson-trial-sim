from pathlib import Path
import pandas as pd
import numpy as np
import simpy
import matplotlib.pyplot as plt

# Get the absolute path to the data directory
current_dir = Path(__file__).parent
data_file = current_dir.parent / 'data' / 'Integrated_Patient_and_Progression_Data.csv'

# Verify file exists
if not data_file.exists():
    raise FileNotFoundError(f"Data file missing at {data_file}")

# CORRECTED PARAMS DEFINITION (using data_file instead of data_path)
params = {
    'beta_slow': -1.071,
    'beta_fast': -0.924,
    'sigma_patient': 35.557,
    'patient_data': pd.read_csv(data_file)  # This was the key fix
}

class ParkinsonTrialDES:
    def __init__(self, env, params):
        self.env = env
        self.params = params
        self.patients = []
        self.results = []
        
        # Model parameters with validation
        self.beta_slow = params.get('beta_slow', -1.071)
        self.beta_fast = params.get('beta_fast', -0.924)
        self.sigma_patient = params.get('sigma_patient', 35.557)
        self.bmi_threshold = 25
        
    def patient_generator(self):
        # Clean and validate patient data
        df = self.params['patient_data'].rename(columns={
            'Patient_ID_x': 'Patient_ID',
            'Treatment_Group_x': 'Treatment_Group'
        }).drop(columns=['Patient_ID_y', 'Treatment_Group_y'], errors='ignore')
        
        # Convert critical columns to numeric
        df['Baseline_UPDRS'] = pd.to_numeric(df['Baseline_UPDRS'], errors='coerce')
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        
        for idx, row in df.iterrows():
            # Calculate responder_status FIRST
    responder_status = self._assign_responder_status(row)
    p = {
        'id': row['Patient_ID'],
        'treatment': row['Treatment_Group'],
        'responder_status': responder_status,  # Use pre-calculated value
        'baseline_updrs': row['Baseline_UPDRS'],
        'progression_rate': self._calculate_progression(row),
        # Pass both row AND responder_status to adherence calculation
        'adherence': self._calculate_adherence(row, responder_status),
        'bmi': row.get('BMI', 25),
        'moca': row.get('MoCA_Score', 26)
    }
    self.patients.append(p)
    self.env.process(self.patient_journey(p))
            
    def _assign_responder_status(self, patient):
        # Enhanced responder criteria with BMI consideration
        if patient['Baseline_UPDRS'] < 30 and patient.get('BMI', 25) < 30:
            return 'High'
        return 'Low'
    
    def _calculate_progression(self, patient):
        # Age-adjusted progression with MoCA moderation
        cognitive_modifier = 1 - (patient.get('MoCA_Score', 26) / 30)
        base_rate = np.random.normal(self.beta_slow, 0.023) if patient['Age'] < 65 \
            else np.random.normal(self.beta_fast, 0.024)
        return base_rate * (1 - 0.4*(patient['Treatment_Group'] == 'Exenatide')) * cognitive_modifier
    
    def _calculate_adherence(self, patient_row, responder_status):
    """Enhanced adherence model with dataset values and adverse effects"""
    base_adherence = patient_row.get('Adherence', 
        0.86 if patient_row['Treatment_Group'] == 'Exenatide' else 0.75)
        
       # Adjust for adverse effects
    if patient_row.get('Nausea', 0) > 3:
        base_adherence *= 0.8
    if patient_row.get('Pancreatitis', 0):
        base_adherence *= 0.5
            
       # Use passed responder_status instead of looking it up
    return min(max(base_adherence * (1 + 0.1*(responder_status == 'High')), 0), 1)
    
    def patient_journey(self, patient):
        costs = {'medication': 0, 'hospitalization': 0, 'monitoring': 0}
        qaly = 0
        updrs = patient['baseline_updrs']
        
        while self.env.now < 96:
            yield self.env.timeout(12)
            
            # Calculate progression with inertia effect
            effective_rate = patient['progression_rate'] * (1/patient['adherence'])
            updrs += effective_rate * (self.env.now/12) * (1 + 0.05*(updrs > 50))
            patient['current_updrs'] = updrs
            
            # Cost calculations
            costs['medication'] += 463 if patient['treatment'] == 'Exenatide' else 117
            costs['monitoring'] += 200 if patient['treatment'] == 'Exenatide' else 50
            
            # QALY calculation with MoCA adjustment
            base_qaly = self._calculate_qaly(updrs)
            moca_modifier = patient.get('MoCA_Score', 26) / 30
            qaly += base_qaly * moca_modifier
            
            # Hospitalization risk with BMI modifier
            bmi_risk = 1.5 if patient.get('BMI', 25) > 30 else 1
            if np.random.rand() < 0.05 * bmi_risk:
                costs['hospitalization'] += 15000
                
            self._record_results(patient, costs, qaly)
    
    def _calculate_qaly(self, updrs):
        # Continuous QALY function
        if updrs < 30: return 0.85 - (updrs/300)
        elif updrs < 50: return 0.65 - ((updrs-30)/200)
        else: return max(0.25, 0.45 - ((updrs-50)/100))
    
    def _record_results(self, patient, costs, qaly):
        self.results.append({
            'week': self.env.now,
            'patient_id': patient['id'],
            'treatment': patient['treatment'],
            'responder_status': patient['responder_status'],
            'updrs': patient['current_updrs'],
            **costs,
            'qaly': qaly,
            'bmi': patient.get('BMI', 25),
            'moca': patient.get('MoCA_Score', 26)
        })



# Simulation execution
env = simpy.Environment()
sim = ParkinsonTrialDES(env, params)
env.process(sim.patient_generator())
env.run(until=96)

# Enhanced Analysis and Visualization
results_df = pd.DataFrame(sim.results)

# 1. UPDRS Progression Plot with Confidence Intervals
plt.figure(figsize=(12,7))
for group in ['Exenatide', 'Placebo']:
    subset = results_df[results_df['treatment'] == group]
    mean = subset.groupby('week')['updrs'].mean()
    std = subset.groupby('week')['updrs'].std()
    plt.plot(mean, label=group)
    plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)
plt.title('UPDRS Progression with Variability Bands')
plt.xlabel('Weeks')
plt.ylabel('UPDRS Score')
plt.legend()

# 2. Cost-Effectiveness Frontier Plot
plt.figure(figsize=(12,7))
treatment_groups = results_df.groupby('treatment')
for name, group in treatment_groups:
    cost = group.groupby('patient_id')['medication'].sum() + group.groupby('patient_id')['hospitalization'].sum()
    effect = group.groupby('patient_id')['qaly'].last()
    plt.scatter(effect, cost, label=name, alpha=0.6)
plt.axhline(y=50000, color='r', linestyle='--', label='$50k/QALY threshold')
plt.title('Cost-Effectiveness Frontier by Treatment')
plt.xlabel('QALYs Gained')
plt.ylabel('Total Costs (USD)')
plt.legend()

# 3. Enhanced Validation Plot
fig, ax = plt.subplots(1, 2, figsize=(15,6))
progression_rates = [p['progression_rate'] for p in sim.patients]
ax[0].hist(progression_rates, bins=30, density=True, alpha=0.7)
ax[0].set_title('Progression Rate Distribution vs Bayesian Posteriors')
ax[0].axvline(x=-1.071, color='r', label='Beta Slow Prior')
ax[0].axvline(x=-0.924, color='g', label='Beta Fast Prior')

adherence_values = [p['adherence'] for p in sim.patients]
ax[1].hist(adherence_values, bins=20, color='purple', alpha=0.7)
ax[1].set_title('Adherence Distribution')
ax[1].set_xlabel('Adherence Rate')

# 4. Enhanced Summary Table with Statistics
summary_table = results_df.groupby('treatment').agg(
    Mean_UPDRS=('updrs', 'mean'),
    Std_UPDRS=('updrs', 'std'),
    Total_Cost=('medication', 'sum'),
    Avg_QALY=('qaly', 'mean'),
    Hospitalization_Rate=('hospitalization', lambda x: (x > 0).mean())
).reset_index()
print("Enhanced Cost-Effectiveness Summary:\n", summary_table)

# 5. BMI-Adjusted Outcomes Analysis
plt.figure(figsize=(12,6))
for status in ['High', 'Low']:
    subset = results_df[results_df['responder_status'] == status]
    plt.scatter(subset['bmi'], subset['qaly'], alpha=0.5, label=status)
plt.title('QALY Outcomes vs BMI by Responder Status')
plt.xlabel('BMI')
plt.ylabel('Final QALY Score')
plt.legend()
plt.show()
