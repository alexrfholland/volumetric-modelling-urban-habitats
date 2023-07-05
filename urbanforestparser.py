import pandas as pd

# Load data from CSV
csv_file = 'data/trees-with-species-and-dimensions-urban-forest.csv'
data = pd.read_csv(csv_file)

# Get unique values in 'Genus' column
unique_genus = data['Genus'].unique()

# Convert to list
unique_genus_list = list(unique_genus)

# Print the list
print(unique_genus_list)

australian_native_genera = [
    "Acacia",
    "Allocasuarina",
    "Angophora",
    "Banksia",
    "Bursaria",
    "Callistemon",
    "Casuarina",
    "Corymbia",
    "Eucalyptus",
    "Grevillea",
    "Hakea",
    "Kunzea",
    "Leptospermum",
    "Lophostemon",
    "Melaleuca",
    "Stenocarpus",
    "Syzygium",
    "Tristaniopsis",
    "Xanthorrhoea"
]

melbourne_indigenous_genera = [
    "Acacia",
    "Allocasuarina",
    "Banksia",
    "Bursaria",
    "Callistemon",
    "Casuarina",
    "Eucalyptus",
    "Grevillea",
    "Hakea",
    "Kunzea",
    "Leptospermum",
    "Melaleuca"
]

