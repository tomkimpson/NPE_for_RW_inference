import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # 2. Neural Posterior Estimation 

    ## What is NPE?

    **Neural Posterior Estimation** is a simulation-based inference method that uses neural networks to learn the posterior distribution p(θ|x) directly from simulated data. Instead of requiring likelihood calculations, NPE:

    1. **Generates training data**: Samples parameters θ from priors, simulates corresponding observations x
    2. **Trains a neural network**: Learns to map observations x to posterior distributions over parameters θ  
    3. **Performs inference**: Given real observed data x_obs, the network outputs p(θ|x_obs)

    ## Sequential NPE (SNPE)

    **Sequential NPE** improves upon standard NPE by iteratively refining the posterior estimate:

    - **Round 1**: Sample from priors, train initial posterior estimate
    - **Round 2+**: Sample from previous posterior estimate, retrain with focused data
    - **Convergence**: Stop when posterior estimates stabilize

    This focuses computational effort on the most relevant parameter regions, leading to more accurate inference.

    ## Running NPE

    Running NPE is typically more compute-intensive than doing a standard linear regression or MCMC. Accordingly we do not run the NPE workflow in this notebook. In general, the workflow can be run by calling `src/main.py` with the relevant command line arguments set. You can see an example of how to run this via a slurm scheduler with some specific arguments in `slurm/run_main.sh`.

    Please see the `README` for some additional notes on running the workflow.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Corner Plot

    Lets see how well our network did at obtaining posteriors on the parameters
    """
    )
    return


@app.cell
def _():
    from pathlib import Path
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np
    import corner
    import scienceplots


    def load_results(results_path):
        print(f"📂 Loading inference results from {results_path}")

        with open(results_path, "rb") as f1:
            results = pickle.load(f1)

        # Extract key components
        posterior_samples = results["posterior_samples"]  # Shape: (5000, 2) for [U, P]
        true_parameters = results["true_parameters"]  # [0.3, 0.7]

        print(f"✅ Successfully loaded results!")
        print(f"📊 Posterior samples shape: {posterior_samples.shape}")
        print(f"🎯 True parameters: U={true_parameters[0]}, P={true_parameters[1]}")

        return posterior_samples, true_parameters


    def create_professional_corner_plot(
        posterior_samples,
        param_names,
        true_parameters,
        nbins=30,
        truth_color="orange",
        savefig_path=None,
        **kwargs,
    ):
        """
        Create a professional, publication-quality corner plot.

        Parameters:
        -----------
        posterior_samples : array-like, shape (n_samples, n_params)
            Posterior samples to plot
        param_names : list of str
            Parameter names for axis labels (LaTeX formatting recommended, e.g., [r'$\theta_1$', r'$\theta_2$'])
        true_parameters : array-like
            True parameter values to mark on the plot
        nbins : int, default 30
            Number of bins for histograms
        truth_color : str, default 'orange'
            Color for truth value lines
        savefig_path : str or None, default None
            Path to save the figure. If None, figure is not saved.
            Will save both .png and .pdf versions if path provided.
        **kwargs : dict
            Additional keyword arguments passed to corner.corner()

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The corner plot figure

        Examples:
        ---------
        >>> samples = np.random.randn(1000, 2)
        >>> param_names = [r'$\theta_1$', r'$\theta_2$']
        >>> true_vals = [0.5, -0.3]
        >>> fig = create_professional_corner_plot(samples, param_names, true_vals,
        ...                                       savefig_path='corner_plot')
        """

        # Set up professional styling
        plt.style.use(["science", "no-latex"])  # Add 'no-latex' if LaTeX not available
        plt.rcParams.update(
            {
                "font.size": 12,
                "axes.linewidth": 1.2,
                "xtick.major.width": 1.2,
                "ytick.major.width": 1.2,
                "xtick.minor.width": 0.8,
                "ytick.minor.width": 0.8,
                "figure.dpi": 100,
                "savefig.dpi": 300,
            }
        )

        # Convert inputs to numpy arrays
        posterior_samples = np.array(posterior_samples)
        true_parameters = np.array(true_parameters)

        print("shapes:", posterior_samples.shape)
        # Professional color scheme
        posterior_color = "#2E86C1"  # Professional blue
        contour_colors = ["#AED6F1", "#5DADE2", "#2E86C1"]  # Gradient blues

        # Default corner plot arguments
        corner_defaults = {
            "labels": param_names,
            "truths": true_parameters,
            "truth_color": truth_color,
            "color": posterior_color,
            "show_titles": True,
            "title_kwargs": {"fontsize": 14, "fontweight": "bold", "pad": 10},
            "label_kwargs": {"fontsize": 16, "fontweight": "bold"},
            "title_fmt": ".3f",
            "bins": nbins,
            "quantiles": [0.16, 0.5, 0.84],  # 68% credible intervals
            "plot_density": True,
            "plot_datapoints": False,  # Clean look without individual points
            "fill_contours": True,
            "contour_kwargs": {"colors": contour_colors, "linewidths": 0.5},
            "hist_kwargs": {"alpha": 0.8, "edgecolor": posterior_color, "linewidth": 1.0},
            "max_n_ticks": 4,
            "use_math_text": True,
            "truth_kwargs": {
                "color": truth_color,
                "linewidth": 2.5,
                "alpha": 0.8,
                "linestyle": "--",
            },
        }

        # Update defaults with any user-provided kwargs
        corner_defaults.update(kwargs)

        # Create the corner plot
        fig = corner.corner(posterior_samples, **corner_defaults)

        # Post-processing improvements
        axes = fig.get_axes()

        # Enhance axis appearance
        for ax in axes:
            if ax is not None:
                # Improve tick appearance
                ax.tick_params(
                    which="major",
                    labelsize=12,
                    width=1.2,
                    length=6,
                    direction="in",
                    top=True,
                    right=True,
                )
                ax.tick_params(
                    which="minor", width=0.8, length=3, direction="in", top=True, right=True
                )

                # Add minor ticks
                ax.minorticks_on()

                # Add subtle grid
                ax.grid(True, alpha=0.3, linewidth=0.5, linestyle=":")

        # Adjust layout
        plt.tight_layout()

        # Manually adjust layout with reduced spacing if spacing is not to your liking
        plt.subplots_adjust(
            hspace=0.05,  # Reduce vertical spacing between subplots
            wspace=0.05,  # Reduce horizontal spacing between subplots
        )

        # Save figure if path provided
        if savefig_path is not None:
            savefig_path = Path(savefig_path)
            # Save PNG version
            png_path = savefig_path.with_suffix(".png")
            fig.savefig(
                png_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
            )

            # Save PDF version for publications
            pdf_path = savefig_path.with_suffix(".pdf")
            fig.savefig(pdf_path, bbox_inches="tight", facecolor="white", edgecolor="none")

            print(f"Figures saved as:\n  - {png_path}\n  - {pdf_path}")

        return fig
    return create_professional_corner_plot, load_results, plt


@app.cell
def _(create_professional_corner_plot, load_results):
    posterior_samples_1D, true_parameters_1D = load_results(
        "notebooks/example_results/results_extracted_1D.pkl"
    )
    create_professional_corner_plot(
        posterior_samples_1D,
        [r"$U$", r"$P$"],
        true_parameters_1D,
        nbins=30,
        truth_color="orange",
        savefig_path="notebooks/images/NPE_corner_plot_1D.png",
    )
    return


@app.cell
def _(create_professional_corner_plot, load_results, plt):
    posterior_samples_2D, true_parameters_2D = load_results(
        "notebooks/example_results/results_extracted_2D.pkl"
    )
    create_professional_corner_plot(
        posterior_samples_2D,
        [r"$U$", r"$P$"],
        true_parameters_2D,
        nbins=30,
        truth_color="orange",
        savefig_path="notebooks/images/NPE_corner_plot_2D.png",
    )
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
