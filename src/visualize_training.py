import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

# ----------------------------------------------- Default Arguments ----------------------------------------------------

batch_size = 1
PATH = 'model.csv'


# ----------------------------------------------- Parsed Arguments -----------------------------------------------------

# Initiate the parser
parser = argparse.ArgumentParser()

# Add long and short argument
parser.add_argument("--path", help="Set path to model file.")

# Read arguments from the command line
args = parser.parse_args()

# Check arguments
print(103*"-")
if args.path:
    PATH = args.path
out = "| PATH: " + PATH
print(out, (100 - len(out))*' ', '|')
print(103*"-")


# ----------------------------------------------- Visualization --------------------------------------------------------
# Read data frame on csv file
df = pd.read_csv(PATH)
sns.set_theme(style="darkgrid")

fig, axs = plt.subplots(nrows=3)
# Plot the loss throughout epochs
sns.lineplot(x="epoch", y="loss_avg", data=df, ax=axs[0])
sns.lineplot(x="epoch", y="lr", data=df, ax=axs[1])
sns.lineplot(x="epoch", y="mAP", data=df, ax=axs[2])
plt.show()
