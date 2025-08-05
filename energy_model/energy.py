__author__ = 'HPC'
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import weibull_min
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.font_manager import FontProperties
import random
from sklearn.metrics import roc_auc_score
import sys
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim * 2)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(input_dim * 2, input_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(input_dim, int(input_dim / 2))
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(int(input_dim / 2), num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x
    
def evaluate_model(y_true, y_pred, average='macro'):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    print(f'Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')
    return acc, precision, recall, f1


def energy_loss(logits, labels, T = 1):
    true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)  
    energy = -true_logits   
    normalization = torch.logsumexp(logits / T, dim=1)  
    loss = energy + normalization
    return loss.mean()  


def energy_ft_loss(outputs, outputs_in, outputs_out, labels): 
    T = 10
    Ec_out = -T * torch.logsumexp(outputs_out / T, dim=1)
    Ec_in = -T * torch.logsumexp(outputs_in / T, dim=1)
    cross_entropy = nn.CrossEntropyLoss()
    m_in = -10     
    m_out = -5     
    loss = cross_entropy(outputs, labels)
    relu = nn.ReLU()
    loss += 0.1 * (torch.pow(relu(Ec_in - m_in), 2).mean() + torch.pow(relu(m_out - Ec_out), 2).mean())
    return loss
        
        
def calculate_energy(logits, T=1):
    energy = -T * torch.logsumexp(logits / T, dim=1)
    return energy
        
def find_intersections(x1, y1, x2, y2):
    """使用插值法找到两个分布的所有交点"""
    f1 = interp1d(x1, y1, kind='linear', fill_value='extrapolate')
    f2 = interp1d(x2, y2, kind='linear', fill_value='extrapolate')
    x_common = np.linspace(max(min(x1), min(x2)), min(max(x1), max(x2)), 500)
    diff = f1(x_common) - f2(x_common)
    intersections = []
    for i in range(1, len(diff)):
        if diff[i - 1] * diff[i] < 0:  
            intersection = x_common[i - 1] + (x_common[i] - x_common[i - 1]) * abs(diff[i - 1] / (diff[i] - diff[i - 1]))
            intersections.append(intersection)
    return intersections


def plot_two_weibull(
    known_energy, known_params, 
    unknown_energy, unknown_params, 
    title, save_path
):
    plt.figure(figsize=(10, 6))
    x_known = np.linspace(0, max(-known_energy), 200)
    pdf_known = weibull_min.pdf(x_known, *known_params)
    plt.plot(x_known, pdf_known, lw=1.5, color=(171 / 255, 191 / 255, 228 / 255), alpha=0.7, label='Known Classes')
    plt.fill_between(x_known, pdf_known, color=(135 / 255, 206 / 255, 250 / 255), alpha=0.3)
    x_unknown = np.linspace(0, max(-known_energy), 200)
    pdf_unknown = weibull_min.pdf(x_unknown, *unknown_params)
    plt.plot(x_unknown, pdf_unknown, lw=1.5, color=(251 / 255, 15 / 255, 15 / 255), alpha=0.9, label='Unknown Classes')
    plt.fill_between(x_unknown, pdf_unknown, color=(251 / 255, 15 / 255, 15 / 255), alpha=0.3)
    known_samples = weibull_min.rvs(*known_params, size=10000)
    unknown_samples = weibull_min.rvs(*unknown_params, size=10000)
    combined_min = min(known_samples.min(), unknown_samples.min())
    combined_max = max(known_samples.max(), unknown_samples.max())
    bins = np.linspace(combined_min, combined_max, 200)  
    plt.hist(known_samples, bins=bins, density=True, alpha=0.3, 
             color=(135 / 255, 206 / 255, 250 / 255),
             edgecolor=(105 / 255, 105 / 255, 105 / 255),
             label='Energy Values of Known Flow', range=(0, max(-known_energy)))
    plt.hist(unknown_samples, bins=bins, density=True, alpha=0.3, 
             color=(251 / 255, 15 / 255, 15 / 255),
             edgecolor=(105 / 255, 105 / 255, 105 / 255), 
             label='Energy Values of Unknown Flow', range=(0, max(-unknown_energy)))
    intersections = find_intersections(x_known, pdf_known, x_unknown, pdf_unknown)
    if intersections:
        for ix in intersections:
            plt.axvline(x=ix, color=(105 / 255, 105 / 255, 105 / 255), linestyle='--', lw=1.5, alpha=1, label=f'threshold τ')
            break
    plt.xlim(0, max(x_known.max(), x_unknown.max()))
    plt.xticks(fontsize=17)  
    plt.yticks(fontsize=17)  
    plt.ylim(0, max(pdf_known.max(), pdf_unknown.max()) * 1.1)
    plt.xlabel('Negative Energy', size=20)
    plt.ylabel('Density', size=20)
    font_prop = FontProperties(size=14)
    plt.legend(prop=font_prop)
    plt.savefig(save_path, dpi=330)
    plt.close()
    print(f"Weibull 分布图已保存到 {save_path}")
