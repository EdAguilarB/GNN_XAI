import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import io
from PIL import Image


def create_training_plot(df, save_path):

    df = pd.read_csv(df)

    epochs = df.iloc[:,0]
    train_loss = df.iloc[:,1]
    #val_loss = df.iloc[:,2]
    test_loss = df.iloc[:,2]

    #min_val_loss_epoch = epochs[val_loss.idxmin()]

    # Create a Matplotlib figure and axis
    plt.figure(figsize=(10, 6), dpi=300)  # Adjust the figure size as needed
    plt.plot(epochs, train_loss, label='Train Loss', marker='o', linestyle='-')
    #plt.plot(epochs, val_loss, label='Validation Loss', marker='o', linestyle='-')
    plt.plot(epochs, test_loss, label='Test Loss', marker='o', linestyle='-')

    #plt.axvline(x=min_val_loss_epoch, color='gray', linestyle='--', label=f'Min Validation Epoch ({min_val_loss_epoch})')

    # Customize the plot
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(False)
    plt.legend()

    # Save the plot in high resolution (adjust file format as needed)
    plt.savefig('{}/loss_vs_epochs.png'.format(save_path), bbox_inches='tight')

    plt.close()


def create_mol_plot(smiles, atom_values, ground_truth):
    # Import necessary modules
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm  # For colormaps
    import io
    from PIL import Image

    # 1. Define the SMILES string and create the molecule
    mol = Chem.MolFromSmiles(smiles)

    # 2. Compute 2D coordinates for visualization
    AllChem.Compute2DCoords(mol)

    # 3. Normalize the atom_values
    min_val = min(atom_values)
    max_val = max(atom_values)
    if max_val - min_val == 0:
        normalized_values = np.zeros_like(atom_values)
    else:
        normalized_values = (atom_values - min_val) / (max_val - min_val)

    # 4. Prepare atom indices and colors
    atom_indices = []
    atom_colors = {}
    highlight_radii = {}

    # Use the 'Reds' colormap for lighter shades
    colormap = cm.get_cmap('Reds')


    # Then, handle ground_truth atoms
    ground_truth_color = (0.6, 0.8, 1.0)  # Light blue color
    for idx in ground_truth:
        atom_indices.append(idx)
        atom_colors[idx] = ground_truth_color
        highlight_radii[idx] = 0.5

    # First, handle atoms from atom_values
    for idx, val in enumerate(normalized_values):
        if val > 0:
            atom_indices.append(idx)
            # Adjust the colormap input to use lighter colors
            rgba = colormap(val * 0.5 + 0.5)  # Shift values towards lighter end
            color = (rgba[0], rgba[1], rgba[2])  # Use RGB components
            atom_colors[idx] = color
            highlight_radii[idx] = 0.5  # Adjust highlight radius if needed



    # Remove duplicate indices
    atom_indices = list(set(atom_indices))

    # 5. Generate the molecule image with atom highlights
    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)  # Image size in pixels
    options = drawer.drawOptions()
    options.addAtomIndices = True  # Display atom indices

    # Draw the molecule with highlights
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=atom_indices,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=highlight_radii
    )
    drawer.FinishDrawing()

    # 6. Display the image without saving to file
    png_data = drawer.GetDrawingText()

    # Display the image using matplotlib
    image = Image.open(io.BytesIO(png_data))
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')  # Hide the axes
    plt.show()


# Import necessary modules
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

def create_mol_plot2(smiles, predicted, ground_truth, save_path):
    # 1. Create the molecule from the SMILES string
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")

    # 2. Compute 2D coordinates for visualization
    AllChem.Compute2DCoords(mol)

    # 3. Determine atom sets and their relationships
    predicted_set = set(predicted)
    ground_truth_set = set(ground_truth)
    intersection = predicted_set & ground_truth_set
    predicted_only = predicted_set - ground_truth_set
    ground_truth_only = ground_truth_set - predicted_set

    # 4. Prepare atom indices and colors
    atom_indices = []
    atom_colors = {}
    highlight_radii = {}

    # Color definitions (colorblind-friendly)
    intersection_color = (27/255, 158/255, 119/255)  # Green
    predicted_only_color = (217/255, 95/255, 2/255)  # Orange
    ground_truth_only_color = (117/255, 112/255, 179/255)  # Purple

    # Atoms in intersection
    for idx in intersection:
        atom_indices.append(idx)
        atom_colors[idx] = intersection_color
        highlight_radii[idx] = 0.5

    # Atoms only in predicted
    for idx in predicted_only:
        atom_indices.append(idx)
        atom_colors[idx] = predicted_only_color
        highlight_radii[idx] = 0.5

    # Atoms only in ground truth
    for idx in ground_truth_only:
        atom_indices.append(idx)
        atom_colors[idx] = ground_truth_only_color
        highlight_radii[idx] = 0.5

    # 5. Generate the molecule image with atom highlights
    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    options = drawer.drawOptions()
    options.addAtomIndices = True

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=atom_indices,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=highlight_radii,
    )
    drawer.FinishDrawing()

    # 6. Display the image without saving to file
    png_data = drawer.GetDrawingText()

    image = Image.open(io.BytesIO(png_data))
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.savefig(save_path)
    plt.close()

