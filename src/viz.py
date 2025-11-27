import os
import matplotlib.pyplot as plt


def save_fig(fig, output_dir, filename, formats=['png']):
    """
    Save a figure in multiple formats.
    """
    os.makedirs(output_dir, exist_ok=True)

    for fmt in formats:
        full_path = os.path.join(output_dir, f"{filename}.{fmt}")

        fig.savefig(
            full_path,
            dpi=300,
            bbox_inches='tight',
            transparent=False,
            facecolor='white'
        )
        print(f"Saved to: {full_path}")

    plt.close(fig)