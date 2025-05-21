import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_ms_dataset(n_samples=5000):
    np.random.seed(42)  # for reproducibility
    
    data = {
        'ID': range(1, n_samples + 1),
        'Age': np.random.randint(20, 61, n_samples),
        'Gender': np.random.choice(['M', 'F'], n_samples),
        'Heart_Rate': np.random.uniform(60, 100, n_samples),
        'Respiration_Rate': np.random.uniform(12, 20, n_samples),
        'Skin_Temperature': np.random.uniform(33, 36, n_samples),
        'Blood_Pulse_Wave': np.random.uniform(60, 100, n_samples),
        'Heart_Rate_Variability': np.random.uniform(20, 70, n_samples),
        'Galvanic_Skin_Response': np.random.uniform(1, 10, n_samples),
        'Energy_Expenditure': np.random.uniform(1500, 2500, n_samples),
        'Steps': np.random.randint(3000, 8001, n_samples),
        'Phone_Lock_Unlock': np.random.uniform(1.5, 5, n_samples),
        'Tapping_Task': np.random.uniform(15, 35, n_samples),
        'Sleep_Duration': np.random.uniform(5, 9, n_samples),
        'MS_Diagnosis': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])  # 40% MS cases
    }
    
    df = pd.DataFrame(data)
    
    # Adjust values for MS patients
    ms_patients = df['MS_Diagnosis'] == 1
    df.loc[ms_patients, 'Heart_Rate'] += np.random.uniform(0, 10, ms_patients.sum())
    df.loc[ms_patients, 'Heart_Rate_Variability'] -= np.random.uniform(0, 15, ms_patients.sum())
    df.loc[ms_patients, 'Galvanic_Skin_Response'] -= np.random.uniform(0, 2, ms_patients.sum())
    df.loc[ms_patients, 'Energy_Expenditure'] -= np.random.uniform(0, 300, ms_patients.sum())
    df.loc[ms_patients, 'Steps'] -= np.random.randint(0, 1501, ms_patients.sum())
    df.loc[ms_patients, 'Phone_Lock_Unlock'] += np.random.uniform(0, 1, ms_patients.sum())
    df.loc[ms_patients, 'Tapping_Task'] -= np.random.uniform(0, 5, ms_patients.sum())
    
    return df

def save_dataset(df, filename='ms_dataset.csv'):
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")

def visualize_data(df):
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    
    # Heart Rate Distribution
    axs[0, 0].hist(df[df['MS_Diagnosis'] == 0]['Heart_Rate'], alpha=0.5, label='Non-MS')
    axs[0, 0].hist(df[df['MS_Diagnosis'] == 1]['Heart_Rate'], alpha=0.5, label='MS')
    axs[0, 0].set_title('Heart Rate Distribution')
    axs[0, 0].legend()
    
    # Steps vs Energy Expenditure
    axs[0, 1].scatter(df[df['MS_Diagnosis'] == 0]['Steps'], df[df['MS_Diagnosis'] == 0]['Energy_Expenditure'], alpha=0.5, label='Non-MS')
    axs[0, 1].scatter(df[df['MS_Diagnosis'] == 1]['Steps'], df[df['MS_Diagnosis'] == 1]['Energy_Expenditure'], alpha=0.5, label='MS')
    axs[0, 1].set_title('Steps vs Energy Expenditure')
    axs[0, 1].set_xlabel('Steps')
    axs[0, 1].set_ylabel('Energy Expenditure')
    axs[0, 1].legend()
    
    # Age Distribution
    axs[1, 0].hist(df[df['MS_Diagnosis'] == 0]['Age'], alpha=0.5, label='Non-MS')
    axs[1, 0].hist(df[df['MS_Diagnosis'] == 1]['Age'], alpha=0.5, label='MS')
    axs[1, 0].set_title('Age Distribution')
    axs[1, 0].legend()
    
    # Tapping Task vs Phone Lock/Unlock Time
    axs[1, 1].scatter(df[df['MS_Diagnosis'] == 0]['Tapping_Task'], df[df['MS_Diagnosis'] == 0]['Phone_Lock_Unlock'], alpha=0.5, label='Non-MS')
    axs[1, 1].scatter(df[df['MS_Diagnosis'] == 1]['Tapping_Task'], df[df['MS_Diagnosis'] == 1]['Phone_Lock_Unlock'], alpha=0.5, label='MS')
    axs[1, 1].set_title('Tapping Task vs Phone Lock/Unlock Time')
    axs[1, 1].set_xlabel('Tapping Task (taps in 10 seconds)')
    axs[1, 1].set_ylabel('Phone Lock/Unlock Time (seconds)')
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('ms_dataset_visualization.png')
    print("Visualization saved to ms_dataset_visualization.png")

if __name__ == "__main__":
    df = generate_ms_dataset()
    save_dataset(df)
    visualize_data(df)
    print("Dataset generation and visualization complete.")