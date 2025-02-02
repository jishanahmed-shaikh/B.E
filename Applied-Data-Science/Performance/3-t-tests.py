import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 1. One Sample T-Test
# Create Reliance data mart dataset
rice_sample = np.array([2.5, 2.7, 2.8, 2.6, 2.4, 2.9, 2.5, 2.6, 2.7, 2.8])
population_mean = 2.5

def perform_one_sample_ttest(sample, pop_mean):
    print("ONE SAMPLE T-TEST")
    print("-" * 50)
    t_stat, p_value = stats.ttest_1samp(sample, pop_mean)
    
    print(f"Sample Mean: {np.mean(sample):.2f}")
    print(f"Population Mean: {pop_mean}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.hist(sample, bins=10, density=True, alpha=0.7)
    plt.axvline(pop_mean, color='red', linestyle='dashed', label='Population Mean')
    plt.axvline(np.mean(sample), color='green', linestyle='dashed', label='Sample Mean')
    plt.title('Distribution of Rice Bag Weights')
    plt.legend()
    plt.show()

# 2. Paired Sample T-Test
# Create pre-post score dataset
pre_scores = np.array([75, 70, 85, 80, 65, 75, 80, 90, 70, 80])
post_scores = np.array([85, 80, 95, 85, 75, 85, 90, 95, 80, 85])

def perform_paired_ttest(pre_data, post_data):
    print("\nPAIRED SAMPLE T-TEST")
    print("-" * 50)
    t_stat, p_value = stats.ttest_rel(pre_data, post_data)
    
    print(f"Pre-test Mean: {np.mean(pre_data):.2f}")
    print(f"Post-test Mean: {np.mean(post_data):.2f}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.boxplot([pre_data, post_data], labels=['Pre-test', 'Post-test'])
    plt.title('Pre vs Post Test Scores')
    plt.ylabel('Scores')
    plt.show()

# 3. Create Crocin dataset (temperature before and after)
temp_before = np.array([101.2, 102.1, 101.8, 102.3, 101.9, 102.0, 101.5, 102.2, 101.7, 102.4])
temp_after = np.array([99.8, 100.1, 99.5, 100.2, 99.9, 99.7, 99.4, 100.0, 99.6, 100.3])

def perform_crocin_analysis(before_data, after_data):
    print("\nCROCIN EFFECTIVENESS T-TEST")
    print("-" * 50)
    t_stat, p_value = stats.ttest_rel(before_data, after_data)
    
    print(f"Mean Temperature Before: {np.mean(before_data):.2f}")
    print(f"Mean Temperature After: {np.mean(after_data):.2f}")
    print(f"Temperature Reduction: {np.mean(before_data - after_data):.2f}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.plot(['Before', 'After'], [np.mean(before_data), np.mean(after_data)], 'bo-')
    plt.title('Average Temperature Before and After Crocin')
    plt.ylabel('Temperature (°F)')
    plt.show()

# Run all analyses
perform_one_sample_ttest(rice_sample, population_mean)
perform_paired_ttest(pre_scores, post_scores)
perform_crocin_analysis(temp_before, temp_after)

# Additional visualization for all three tests combined
plt.figure(figsize=(15, 5))

# One-sample t-test
plt.subplot(1, 3, 1)
plt.hist(rice_sample, bins=10, alpha=0.7)
plt.axvline(population_mean, color='red', linestyle='dashed')
plt.title('Rice Bag Weights')

# Paired t-test (scores)
plt.subplot(1, 3, 2)
plt.scatter(pre_scores, post_scores)
plt.plot([60, 95], [60, 95], 'r--')
plt.xlabel('Pre-scores')
plt.ylabel('Post-scores')
plt.title('Pre vs Post Scores')

# Paired t-test (temperature)
plt.subplot(1, 3, 3)
plt.boxplot([temp_before, temp_after], labels=['Before', 'After'])
plt.title('Temperature Before/After Crocin')

plt.tight_layout()
plt.show()
