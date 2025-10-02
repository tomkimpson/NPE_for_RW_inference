import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():

    """
    Import required modules
    """
    import sys
    import os


    # Add src directory to path
    sys.path.append(os.path.join(os.getcwd(), 'src')) # Assumes notebook launched from project root



    import matplotlib.pyplot as plt
    import scienceplots  # Required for the 'science' style to be available
    from predict import load_prediction_results,compute_prediction_intervals,plot_prediction_intervals


    prediction_data_1D = load_prediction_results('results/workflow_20250827_224139/inference_results/predictive_results.pkl')
    prediction_data_2D = load_prediction_results('results/workflow_20250919_083626/inference_results/predictive_results.pkl')



    def generate_posterior_predictive_plot(data,save_path):

        # Extract
        prediction_results = data['prediction_results']
        input_metadata = data['input_metadata']
        observed_data = data.get('observed_data', None)

        # Extract simulation parameters from loaded metadata
        sim_params = input_metadata['simulation_params']
        actual_Lx = sim_params['Lx']
        actual_Ly = sim_params['Ly']
        actual_T = sim_params['T']

        print(f"⚙️  Using simulation settings from loaded data: Lx={actual_Lx}, Ly={actual_Ly}, T={actual_T}")

        #Recompute prediction intervals with a smoothing kernel (optional)
        raw_predictions = prediction_results['predictions'] 
        prediction_results = compute_prediction_intervals(
                        predictions=raw_predictions,
                        percentiles=prediction_results['metadata']['percentiles'],
                        smooth_sigma=1.0)


        # Main prediction plot
        fig2 = plot_prediction_intervals(
            prediction_results=prediction_results,
            observed_data=observed_data,
            Lx=actual_Lx
        )
        plt.savefig(save_path,dpi=300, bbox_inches='tight')
        plt.show()
    return (
        generate_posterior_predictive_plot,
        plt,
        prediction_data_1D,
        prediction_data_2D,
    )


@app.cell
def _(generate_posterior_predictive_plot, prediction_data_1D):
    generate_posterior_predictive_plot(prediction_data_1D,'notebooks/images/example_predictive_plot_for_paper_1D.png')
    return


@app.cell
def _(generate_posterior_predictive_plot, prediction_data_2D):
    generate_posterior_predictive_plot(prediction_data_2D,'notebooks/images/example_predictive_plot_for_paper_2D.png')
    return


@app.cell
def _(plt):

    from simulator import RandomWalkSimulator
    import numpy as np
    from scipy.ndimage import gaussian_filter

    def generate_posterior_predictive_plot_2D(data, save_path):
        # Set up enhanced professional styling for publication quality
        plt.rcParams.update({
            "font.size": 14,
            "axes.linewidth": 1.5,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.minor.width": 1.0,
            "ytick.minor.width": 1.0,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "legend.fontsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.numpoints": 1,
            "legend.handlelength": 2.5,
            "legend.handletextpad": 0.8,
            "legend.columnspacing": 1.0,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold"
        })

        # Create single figure with contour plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Extract
        prediction_results = data['prediction_results']
        input_metadata = data['input_metadata']
        # Extract simulation parameters
        sim_params = input_metadata['simulation_params']
        actual_Lx = sim_params['Lx']
        actual_Ly = sim_params['Ly']
        actual_T = sim_params['T']
        print(f"⚙️  Using simulation settings from loaded data: Lx={actual_Lx}, Ly={actual_Ly}, T={actual_T}")

        # Extract 2D predictions (shape: n_samples × Ly × Lx)
        predictions_2d = prediction_results['predictions']
        print(f"📊 Predictions shape: {predictions_2d.shape}")

        prob_map1 = predictions_2d.mean(axis=0)  # Shape: (Ly, Lx)
        prob_map = gaussian_filter(prob_map1, sigma=1.0)

        # Generate observed 2D data
        simulator = RandomWalkSimulator(
            Lx=actual_Lx,
            Ly=actual_Ly,
            initial_region_half_width=25
        )
        _, _, obs_final_positions = simulator.simulate(
            U=0.3, P=0.7, T=actual_T, random_seed=42+1000, use_2d_output=False
        )

        # Check the range of probabilities
        print(f"Min probability: {prob_map.min()}")
        print(f"Max probability: {prob_map.max()}")
        print(f"Mean probability: {prob_map.mean()}")
        print(f"Cells with p > 0.5: {(prob_map > 0.5).sum()}")
        print(f"Cells with p > 0.1: {(prob_map > 0.1).sum()}")

        # Create centered coordinate system for the heatmap
        x_min = -(actual_Lx // 2)
        x_max = actual_Lx // 2
        y_min = 0
        y_max = actual_Ly

        # Enhanced professional color scheme
        orange = "#E67E22"
        dark_gray = "#2C3E50"

        # Use a colormap that makes zero values very pale/white
        cmap = 'Blues'
        # Plot with extent to set correct coordinate system
        im = ax.imshow(prob_map, cmap=cmap, vmin=0, vmax=0.35, aspect='auto',
                       extent=[x_min, x_max, y_min, y_max], origin='lower')

        # Contours will now also use the centered coordinates
        threshold_high = 0.20
        threshold_mid = 0.10
        threshold_low = 0.05

        # Need to create coordinate arrays for contour
        x_coords = np.linspace(x_min, x_max, actual_Lx)
        y_coords = np.linspace(y_min, y_max, actual_Ly)

        # Enhanced contours with professional styling
        contours = ax.contour(x_coords, y_coords, prob_map,
                              levels=[threshold_low, threshold_mid, threshold_high],
                              colors=dark_gray, linewidths=[1.5, 2.0, 2.5],
                              linestyles=['dotted', 'dashed', 'solid'], alpha=0.8,zorder=20)
        ax.clabel(contours, inline=True, fontsize=16, fmt='%.2f')


        # Enhanced colorbar with professional styling
        cbar = plt.colorbar(im, label='Probability of Occupation')
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('Probability of Occupation', fontsize=16, fontweight='bold')

        # Enhanced scatter plot for observed data with better visibility
        xobs, yobs = zip(*obs_final_positions)
        ax.scatter(xobs, yobs, c=orange, s=60, marker='o',
                   label='Observed Data', zorder=10, alpha=0.95,
                   edgecolors='white', linewidth=2.0)

        # Enhanced professional axis formatting with larger, bolder labels
        ax.set_xlabel('x', fontsize=18, fontweight='bold', labelpad=12)
        ax.set_ylabel('y', fontsize=18, fontweight='bold', labelpad=12)

        # Enhanced professional grid
        ax.grid(True, alpha=0.4, linewidth=0.6, linestyle=":", color='gray')
        ax.set_axisbelow(True)

        # Enhanced professional tick styling
        ax.tick_params(
            which="major",
            labelsize=14,
            width=1.5,
            length=8,
            direction="in",
            top=True,
            right=True,
            pad=8
        )
        ax.tick_params(
            which="minor",
            width=1.0,
            length=4,
            direction="in",
            top=True,
            right=True
        )
        ax.minorticks_on()
        #ax.set_ylim(-1,51)

        # Enhanced legend with professional styling
        legend = ax.legend(
            loc='upper right',
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.95,
            edgecolor='darkgray',
            facecolor='white',
            handlelength=3.0,
            handletextpad=1.0,
            columnspacing=1.2,
            numpoints=1,
            markerscale=1.2,
            markerfirst=True,
            fontsize=14,
        )
        legend.get_frame().set_linewidth(1.0)

        # Enhanced professional layout
        plt.tight_layout(pad=2.0)

        # Save high-quality figures for publication
        # Save PNG version
        png_path = save_path
        fig.savefig(
            png_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none'
        )

        # Save PDF version for publications
        pdf_path = save_path.replace('.png', '.pdf')
        fig.savefig(
            pdf_path, bbox_inches='tight',
            facecolor='white', edgecolor='none'
        )

        print(f"✅ Publication-quality figures saved as:\n  - {png_path}\n  - {pdf_path}")
        plt.show()
    return (generate_posterior_predictive_plot_2D,)


@app.cell
def _(generate_posterior_predictive_plot_2D, prediction_data_2D):
    generate_posterior_predictive_plot_2D(prediction_data_2D,'notebooks/images/example_predictive_plot_for_paper_2D.png')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
