import re
import numpy as np
import seaborn as sns
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def visualize(log_path):
  loss_pattern = re.compile(r'\[epoch:\s*(\d+), iter:\s*([\d,]+),.*?] .*?l_pix: ([\-\d\.e\+]+)')
  # validation_pattern = re.compile(r'Validation .*?psnr: ([\d\.]+).*?ssim: ([\d\.]+)')
  validation_pattern = re.compile(r'\[epoch:\s*(\d+), iter:\s*([\d,]+),.*?] m_psnr: ([\d\.e\+]+) m_ssim: ([\d\.e\+]+)')
  
  loss_records = []
  validation_records = []

  with open(log_path, 'r') as f:
    for line in f:
      loss_match = loss_pattern.search(line)
      if loss_match:
        epoch = int(loss_match.group(1))
        iter_count = int(loss_match.group(2).replace(',', ''))
        loss = float(loss_match.group(3))
        loss_records.append({'epoch': epoch, 'iter': iter_count, 'loss': loss})
        continue
      validation_match = validation_pattern.search(line)
      if validation_match:
        epoch = int(validation_match.group(1))
        iter_count = int(validation_match.group(2).replace(',', ''))
        psnr = float(validation_match.group(3))
        # ssim = float(validation_match.group(4))
        # validation_records.append({'epoch': epoch, 'iter': iter_count, 'psnr': psnr, 'ssim': ssim})
        validation_records.append({'epoch': epoch, 'iter': iter_count, 'psnr': psnr})
        continue
  
  loss_df = pd.DataFrame(loss_records)
  validation_df = pd.DataFrame(validation_records)
  
  # sns.lineplot(x=loss_df['iter'], y=loss_df['loss'], label='Loss', marker='o')

  # Validation PSNR Plot
  sns.lineplot(x=validation_df['iter'], y=validation_df['psnr'], label='Validation PSNR', marker='s')

  # Plot 꾸미기
  # plt.title('Iteration vs Loss and Validation PSNR')
  plt.xlabel('Iteration')
  plt.ylabel('Value')
  plt.legend()
  plt.grid(True)
  # plt.show()
  plt.savefig('result.png', dpi=200)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--log_path')
  args = parser.parse_args()
  visualize(args.log_path)