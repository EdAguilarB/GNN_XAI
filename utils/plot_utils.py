import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.metrics import ConfusionMatrixDisplay


def create_training_plot(df, save_path):

    df = pd.read_csv(df)

    epochs = df.iloc[:, 0]
    train_loss = df.iloc[:, 2]
    val_loss = df.iloc[:, 3]
    test_loss = df.iloc[:, 4]

    min_val_loss_epoch = epochs[val_loss.idxmin()]

    # Create a Matplotlib figure and axis
    plt.figure(figsize=(10, 6), dpi=300)  # Adjust the figure size as needed
    plt.plot(epochs, train_loss, label="Train Loss", marker="o", linestyle="-")
    plt.plot(epochs, val_loss, label="Validation Loss", marker="o", linestyle="-")
    plt.plot(epochs, test_loss, label="Test Loss", marker="o", linestyle="-")

    plt.axvline(
        x=min_val_loss_epoch,
        color="gray",
        linestyle="--",
        label=f"Min Validation Epoch ({min_val_loss_epoch})",
    )

    # Customize the plot
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(False)
    plt.legend()

    # Save the plot in high resolution (adjust file format as needed)
    plt.savefig("{}/loss_vs_epochs.png".format(save_path), bbox_inches="tight")

    plt.close()


def create_parity_plot(df, save_path):
    """
    Creates a high-quality parity plot with points colored by dataset type (train, test, validation).

    Parameters:
        df (pd.DataFrame): A DataFrame containing 'y_true', 'y_pred', and 'set' columns.
        save_path (str): Path to save the parity plot.

    Returns:
        None
    """
    # Set up the plot aesthetics using Seaborn
    sns.set(style="whitegrid")

    # Create the figure
    plt.figure(figsize=(8, 8), dpi=300)

    # Plot all points, colored by the 'set' column
    sns.scatterplot(
        data=df,
        x="y_true",
        y="y_pred",
        hue="set",
        palette={"train": "blue", "test": "orange", "val": "green"},
        alpha=0.6,
        s=50,  # Adjust marker size
        edgecolor="black",
    )

    # Add the identity line
    min_val = min(df["y_true"].min(), df["y_pred"].min())
    max_val = max(df["y_true"].max(), df["y_pred"].max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Identity Line",
    )

    # Customize plot appearance
    plt.xlabel("True Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.title("Parity Plot", fontsize=14)
    plt.legend(title="Set", fontsize=10, title_fontsize=12)
    plt.grid(True)

    # Save the plot in high resolution
    plt.savefig(f"{save_path}/parity_plot.png", bbox_inches="tight")

    # Close the plot to avoid overlapping figures
    plt.close()
    plt.clf()


def plot_confusion_matrix(
    y_true, y_pred, classes=None, title="Confusion matrix", save_path=None
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, display_labels=classes, cmap=plt.cm.Blues
    )
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format="d")

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, fontweight="bold")

    for text in ax.texts:
        text.set_fontsize(16)  # Increase number size
        text.set_fontweight("bold")  # Make numbers bold

    if title is not None:
        plt.title(title, fontsize=16, fontweight="bold")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    else:
        plt.show()

    plt.close()
    plt.clf()


def create_mol_plot(smiles, atom_values, ground_truth):

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
    colormap = cm.get_cmap("Reds")

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
        highlightAtomRadii=highlight_radii,
    )
    drawer.FinishDrawing()

    # 6. Display the image without saving to file
    png_data = drawer.GetDrawingText()

    # Display the image using matplotlib
    image = Image.open(io.BytesIO(png_data))
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")  # Hide the axes
    plt.show()


import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# Import necessary modules
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D


def create_mol_plot2(smiles, predicted, ground_truth=None, save_path=None):
    """
    Create a molecular plot highlighting predicted and ground truth atom sets.

    Parameters:
        smiles (str): SMILES string of the molecule.
        predicted (list[int]): List of atom indices predicted to be important.
        ground_truth (list[int] or None): List of ground truth atom indices (optional).
        save_path (str or None): Path to save the plot. If None, the plot is not saved.

    Returns:
        None
    """
    # 1. Create the molecule from the SMILES string
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")

    # 2. Compute 2D coordinates for visualization
    AllChem.Compute2DCoords(mol)

    # 3. Determine atom sets and their relationships
    predicted_set = set(predicted)
    ground_truth_set = set(ground_truth) if ground_truth is not None else set()
    intersection = predicted_set & ground_truth_set
    predicted_only = predicted_set - ground_truth_set
    ground_truth_only = ground_truth_set - predicted_set

    # 4. Prepare atom indices and colors
    atom_indices = []
    atom_colors = {}
    highlight_radii = {}

    # Color definitions (colorblind-friendly)
    intersection_color = (27 / 255, 158 / 255, 119 / 255)  # Green
    predicted_only_color = (217 / 255, 95 / 255, 2 / 255)  # Orange
    ground_truth_only_color = (117 / 255, 112 / 255, 179 / 255)  # Purple

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
    if ground_truth is not None:
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

    # Save the image if a save_path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.close()
    plt.clf()


def draw_molecule_with_similarity_map(smiles, weights, save_path=None):
    """
    Draws a molecule with a heatmap overlay using RDKit's SimilarityMaps.

    Parameters:
    - smiles (str): SMILES string of the molecule.
    - weights (list or np.array): Importance weights per atom.
    - save_path (str, optional): If provided, saves the image to this path.

    Returns:
    - Displays the molecule heatmap using matplotlib.
    """

    # Convert SMILES to molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")

    # Compute 2D coordinates for visualization
    AllChem.Compute2DCoords(mol)

    # Normalize weights
    weights = np.array(weights)
    # min_w, max_w = weights.min(), weights.max()
    # norm_weights = (weights - min_w) / (max_w - min_w)  # Normalize to [0,1]

    # Generate similarity map
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol, weights, colorMap="coolwarm", alpha=0.6
    )

    plt.gca().set_facecolor("white")  # Ensures axes background is also white

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.clf()  # Clear the plot for the next molecule
    plt.close()
