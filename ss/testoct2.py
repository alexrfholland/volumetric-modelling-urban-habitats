import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Choose the color map (e.g., "viridis")
color_map = cm.get_cmap('plasma')

positions = [.1,.4,.7,0.8, .99]

colours = []

for pos in positions:
    col = color_map(pos)
    colours.append(col)
    print(f"Color at {pos}: {col}")

# Plot a colorbar to visualize the color map
plt.imshow(colours, cmap=color_map)
plt.colorbar()
plt.show()
